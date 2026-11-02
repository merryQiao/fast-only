"""Prepare a bounded, log-disjoint NAVSIM mini cohort without resizing JPEGs.

Run ``python -m tools.prepare_navsim_smoke --config configs/navsim_preflight.yaml``
to rebuild from retained files, or add ``--acquire`` for bounded public downloads.
The network budget includes failed/short response bodies consumed by the reader.
Archive LFS identities are provenance, not a claim that a partial archive passed
the full LFS checksum. Local retained files have independent SHA256 hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import tarfile

from fast_navsim.acquire import (ByteBudget, HTTPRangeReader, download_prefix_ranges,
                                 extract_tar_prefix, fetch_bytes, write_json)
from fast_navsim.data import (available_window_indices, build_window, split_logs,
                              write_manifests)

NAVSIM_REVISION = "0a380a9063d7162ec93d0f51e9990ebac585f720"
HF_REPOSITORY = "OpenDriveLab/OpenScene"
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_audit(destination,source_directory,retained_files,revision,network_bytes,
                       archive_identities=None):
    """Write immutable upstream identities and verify every retained local hash."""
    source_directory=Path(source_directory).resolve();sources=[]
    identity=json.loads((source_directory/"source_identity.json").read_text())
    for item in identity["files"]:
        sources.append({"path":str(source_directory/item["source_path"]),
                        "sha256":item["sha256"],"url":item["url"],
                        "revision":identity["navsim_revision"],"kind":"official_source"})
    extra=source_directory/"extra_source_acquisition.json"
    if extra.exists():sources.extend(json.loads(extra.read_text())["sources"])
    sources.extend(retained_files)
    unique={}
    for item in sources:
        path=str(Path(item["path"]).resolve())
        if sha256_file(path)!=item["sha256"]:
            raise ValueError(f"Retained source SHA256 mismatch: {path}")
        unique[path]=dict(item,path=path)
    report={"navsim_revision":identity["navsim_revision"],"openscene_revision":revision,
            "sources":list(unique.values()),"network_response_body_bytes_charged":network_bytes,
            "archive_identities":archive_identities or [],"full_archives_verified":False,
            "protocol":"bounded public mini smoke; no official navtrain/navtest score",
            "coordinate_frame":"current rear axle SE2; right-handed; no CARLA y flip",
            "command_policy":"preserve native four-way one-hot index; semantic names unverified",
            "timestamps":"actual microseconds; 500000us intervals with +/-1000us tolerance",
            "targets":"8 future [x,y,heading] at .5s intervals; quaternion yaw matches pinned pyquaternion",
            "image_policy":"original front CAM_F0 JPEG bytes; no resize"}
    write_json(destination,report)
    return report


def build_from_retained(records, sensor_root, destination, seed=42,
                        train_windows=6, val_windows=4, revision=None,
                        data_root=None, network_bytes=0):
    """Build manifests from selected complete logs and retained history JPEGs.

    records maps log name to frames, metadata_path, metadata_sha256 and optional
    camera_archive. Log assignment is fixed before checking available windows.
    Only TRAIN rows contribute normalization statistics.
    """
    splits=split_logs(records,2,1,seed=seed)
    sensor_root=Path(sensor_root).resolve()
    available={str(path.relative_to(sensor_root)) for path in sensor_root.rglob("*.jpg")}
    rows={split:[] for split in splits}
    for split,names in splits.items():
        for name in names:
            record=records[name];frames=record["frames"]
            source={"navsim_revision":NAVSIM_REVISION,"openscene_revision":revision,
                    "upstream_split":"mini","metadata_path":record["metadata_path"],
                    "metadata_sha256":record["metadata_sha256"],
                    "camera_archive":record.get("camera_archive")}
            limit=train_windows if split=="train" else val_windows
            candidates=[]
            for index in available_window_indices(frames,available):
                try:candidates.append(build_window(frames,index,split,sensor_root,source))
                except ValueError:continue
                if len(candidates)>=limit:break
            if len(candidates)<limit:
                raise ValueError(f"Need {limit} valid retained windows for {split} log {name}; found {len(candidates)}")
            rows[split].extend(candidates)
    provenance={"navsim_revision":NAVSIM_REVISION,"openscene_revision":revision,
                "upstream_split":"mini","split_seed":seed,
                "data_root":str(Path(data_root or sensor_root).resolve()),
                "network_response_body_bytes":network_bytes,
                "full_archives_verified":False,
                "command_index_order":"native four-way one-hot; semantic names unverified",
                "timestamp_tolerance_us":1000,"normalization":"TRAIN only",
                "image_policy":"original JPEG; no resize"}
    report=write_manifests(rows,destination,provenance)
    report.update(split_seed=seed,data_root=provenance["data_root"])
    write_json(Path(destination)/"build_report.json",report)
    return report


def extract_complete_metadata(prefix,destination):
    """Retain only complete pickle members from a deliberately partial tar.gz."""
    destination=Path(destination).resolve();retained=[]
    try:
        with tarfile.open(prefix,"r|gz") as archive:
            for member in archive:
                path=PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe metadata path {member.name}")
                if not member.isfile() or not member.name.endswith(".pkl"):continue
                try:blob=archive.extractfile(member).read()
                except (EOFError,tarfile.ReadError):break
                if len(blob)!=member.size:break
                # The archive is pinned to the public OpenScene mini revision.
                frames=pickle.loads(blob)
                if not isinstance(frames,list) or not frames:continue
                target=destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
                retained.append({"path":str(target),"member":member.name,"bytes":len(blob),
                                 "sha256":hashlib.sha256(blob).hexdigest()})
    except (EOFError,tarfile.ReadError):
        pass
    return retained


def candidate_records(retained,camera_index):
    """Select first-in-camera-shard logs in metadata order, before windows."""
    records={}
    for item in retained:
        if sha256_file(item["path"])!=item["sha256"]:
            raise ValueError("Retained metadata SHA256 mismatch")
        with Path(item["path"]).open("rb") as stream:frames=pickle.load(stream)
        name=frames[0]["log_name"]
        matches=[key for key,names in camera_index.items() if names[0]==name]
        if not matches:continue
        shard=matches[0].rsplit("_",1)[1]
        records[name]={"frames":frames,"metadata_path":item["path"],
                       "metadata_sha256":item["sha256"],"shard":shard,
                       "camera_archive":f"openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{shard}.tgz"}
        if len(records)==3:break
    return records


def acquire_camera(record,name,split,sensor_dest,budget,revision,output,limit,
                   archive_limit=225_000_000):
    frames=record["frames"];sensor_root=Path(sensor_dest)/"openscene-v1.1/sensor_blobs/mini"
    bypath={frame["cams"]["CAM_F0"]["data_path"]:i for i,frame in enumerate(frames)}
    available={str(path.relative_to(sensor_root)) for path in sensor_root.rglob("*.jpg")}
    valid={}
    for index in range(3,len(frames)-8):
        try:valid[index]=build_window(frames,index,split,sensor_root)
        except ValueError:continue
    indices={bypath[path] for path in available if path in bypath}
    def count():return sum(all(i-d in indices for d in range(4)) for i in valid)
    if count()>=limit:return None
    url=f"https://huggingface.co/datasets/{HF_REPOSITORY}/resolve/{revision}/{record['camera_archive']}"
    transport=HTTPRangeReader(url,budget,archive_limit,chunk_size=2_000_000)
    def stop(retained):
        relative=retained[-1]["member"].split("sensor_blobs/mini/",1)[1]
        if relative in bypath:indices.add(bypath[relative])
        if len(retained)%25==0:
            print(json.dumps({"log":name,"retained_front_images":len(retained),
                              "available_windows":count(),"network_bytes":budget.used}),flush=True)
        return count()>=limit
    result=extract_tar_prefix(transport,sensor_dest,ByteBudget(archive_limit+1_000_000),
                              lambda p:f"/{name}/CAM_F0/" in p and p.endswith(".jpg"),
                              len(frames),archive_limit+1_000_000,stop)
    result.update(url=url,archive_path=record["camera_archive"],
                  response_body_bytes=transport.response_body_bytes,
                  response_prefix_sha256=transport.sha.hexdigest(),requests=transport.requests)
    write_json(Path(output)/f"camera_{record['shard']}_acquisition.json",result)
    if count()<limit:raise RuntimeError(f"Insufficient windows within bounded camera prefix: {name}")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/navsim_preflight.yaml")
    parser.add_argument("--acquire",action="store_true")
    parser.add_argument("--max-network-bytes",type=int,default=1_000_000_000)
    parser.add_argument("--archive-limit",type=int,default=225_000_000)
    parser.add_argument("--train-windows",type=int,default=6)
    parser.add_argument("--val-windows",type=int,default=4)
    args=parser.parse_args(argv)
    import yaml
    config_path=Path(args.config)
    if not config_path.is_absolute():config_path=REPO_ROOT/config_path
    cfg=yaml.safe_load(config_path.read_text());seed=int(cfg["split_seed"])
    def absolute(path):
        path=Path(path);return path.resolve() if path.is_absolute() else (REPO_ROOT/path).resolve()
    root=absolute(cfg["data"]["root"])
    output=absolute(cfg["data"]["train_manifest"]).parent
    audit=absolute(cfg["data"]["source_audit"]).parent/"source"
    audit.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    budget_path=audit/"network_budget.json"
    prior=json.loads(budget_path.read_text()) if budget_path.exists() else {"used":0}
    budget=ByteBudget(args.max_network_bytes,prior["used"])
    try:
        api=audit/"hf_dataset_api.json"
        if not api.exists():
            if not args.acquire:raise ValueError("Source evidence absent; use --acquire")
            api.write_bytes(fetch_bytes(f"https://huggingface.co/api/datasets/{HF_REPOSITORY}",budget))
        revision=json.loads(api.read_text())["sha"]
        if len(revision)!=40:raise ValueError("Dataset revision is not an immutable git SHA")
        def cached(name,url):
            path=audit/name
            if not path.exists():
                if not args.acquire:raise ValueError(f"Missing retained source {name}")
                path.write_bytes(fetch_bytes(url,budget))
            return json.loads(path.read_text())
        base=f"https://huggingface.co/datasets/{HF_REPOSITORY}/resolve/{revision}/"
        camera_index=cached("mini_camera_index.json",base+"openscene-v1.1/openscene_sensor_mini_0-31.json")
        camera_tree=cached("hf_camera_tree.json",f"https://huggingface.co/api/datasets/{HF_REPOSITORY}/tree/{revision}/openscene-v1.1/openscene_sensor_mini_camera?limit=1000")
        cached("hf_openscene_tree.json",f"https://huggingface.co/api/datasets/{HF_REPOSITORY}/tree/{revision}/openscene-v1.1?limit=1000")
        identity=audit/"source_identity.json"
        if not identity.exists():
            if not args.acquire:raise ValueError("Official source evidence absent; use --acquire")
            files=[]
            for path in ("navsim/common/dataclasses.py","navsim/agents/ego_status_mlp_agent.py",
                         "docs/agents.md","navsim/planning/simulation/planner/pdm_planner/utils/pdm_geometry_utils.py"):
                url=f"https://raw.githubusercontent.com/autonomousvision/navsim/{NAVSIM_REVISION}/{path}"
                blob=fetch_bytes(url,budget);target=audit/path
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
                files.append({"source_path":path,"url":url,"sha256":hashlib.sha256(blob).hexdigest(),"bytes":len(blob)})
            write_json(identity,{"navsim_revision":NAVSIM_REVISION,"files":files})
        retained_path=output/"metadata_retained.json"
        retained=json.loads(retained_path.read_text())["files"] if retained_path.exists() else []
        records=candidate_records(retained,camera_index)
        if len(records)<3:
            if not args.acquire:raise ValueError("Need three complete retained metadata logs")
            prefix=root/"metadata_prefix12m.gz"
            for cap in (12_000_000,30_000_000,60_000_000):
                if prefix.exists() and prefix.stat().st_size>cap:continue
                result=download_prefix_ranges(base+"openscene-v1.1/openscene_metadata_mini.tgz",prefix,budget,cap,resume=True)
                write_json(output/f"metadata_prefix_{cap}_acquisition.json",result)
                retained=extract_complete_metadata(prefix,root/"metadata")
                write_json(retained_path,{"files":retained,"full_archive_verified":False})
                records=candidate_records(retained,camera_index)
                if len(records)==3:break
            if len(records)<3:raise RuntimeError("Metadata prefix cap cannot yield three affordable camera logs")
        splits=split_logs(records,2,1,seed=seed)
        write_json(output/"planned_splits.json",{"split_seed":seed,"splits":splits,
                   "selection":"first three complete metadata logs appearing first in camera shards; before windows"})
        sensor_dest=root/"sensors";sensor_root=sensor_dest/"openscene-v1.1/sensor_blobs/mini"
        if args.acquire:
            for split,names in splits.items():
                for name in names:
                    result=acquire_camera(records[name],name,split,sensor_dest,budget,revision,output,
                                           args.train_windows if split=="train" else args.val_windows,args.archive_limit)
                    if result:
                        result["archive_identity"]=next(item for item in camera_tree if item["path"]==records[name]["camera_archive"])
                        write_json(output/f"camera_{records[name]['shard']}_acquisition.json",result)
        report=build_from_retained(records,sensor_root,output,seed,args.train_windows,
                                    args.val_windows,revision,root,budget.used)
        retained_sources=[];archive_identities=[]
        metadata_url=base+"openscene-v1.1/openscene_metadata_mini.tgz"
        for item in retained:
            retained_sources.append(dict(item,url=metadata_url,revision=revision,kind="retained_metadata"))
        needed={image for split in ("train","val","test")
                for line in (output/f"{split}.jsonl").read_text().splitlines()
                for image in json.loads(line)["images"]}
        for result_path in sorted(output.glob("camera_*_acquisition.json")):
            result=json.loads(result_path.read_text())
            if "archive_identity" in result:archive_identities.append(result["archive_identity"])
            kept=[]
            for item in result["files"]:
                if item["path"] not in needed:
                    Path(item["path"]).unlink(missing_ok=True)
                else:
                    kept.append(item)
                    retained_sources.append(dict(item,url=result["url"],revision=revision,kind="retained_original_jpeg"))
            if kept!=result["files"]:
                result["discarded_front_images"]=result.get("discarded_front_images",0)+len(result["files"])-len(kept)
                result["files"]=kept;write_json(result_path,result)
        for filename,url in (("hf_dataset_api.json",f"https://huggingface.co/api/datasets/{HF_REPOSITORY}"),
                             ("hf_openscene_tree.json",f"https://huggingface.co/api/datasets/{HF_REPOSITORY}/tree/{revision}/openscene-v1.1"),
                             ("hf_camera_tree.json",f"https://huggingface.co/api/datasets/{HF_REPOSITORY}/tree/{revision}/openscene-v1.1/openscene_sensor_mini_camera"),
                             ("mini_camera_index.json",base+"openscene-v1.1/openscene_sensor_mini_0-31.json")):
            retained_sources.append({"path":str(audit/filename),"sha256":sha256_file(audit/filename),
                                     "url":url,"revision":revision,"kind":"source_api_snapshot"})
        metadata_tree=json.loads((audit/"hf_openscene_tree.json").read_text())
        archive_identities.append(next(item for item in metadata_tree if item["path"]=="openscene-v1.1/openscene_metadata_mini.tgz"))
        write_source_audit(absolute(cfg["data"]["source_audit"]),audit,retained_sources,
                           revision,budget.used,archive_identities)
        # Honor explicit configured destinations even when they are outside the manifest directory.
        for key,filename in (("train_manifest","train.jsonl"),("val_manifest","val.jsonl"),
                             ("test_manifest","test.jsonl"),("stats","stats.json")):
            target=absolute(cfg["data"][key]);origin=output/filename
            if target!=origin:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(origin.read_bytes())
        print(json.dumps(report,indent=2))
        return report
    finally:
        write_json(budget_path,dict(prior,limit=budget.limit,used=budget.used))


if __name__=="__main__":main()
