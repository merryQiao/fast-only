import math

import pytest

from fast_navsim.data import build_window, compute_ego_stats, split_logs, _pose, NavsimDataset


def frames(n=24, log="log-a"):
    return [dict(token=f"{log}-{i}", timestamp=1_600_000_000_000_000+i*500_000,
                 log_name=log, scene_token=f"scene-{log}",
                 ego2global_translation=[10., float(i), 0.],
                 ego2global_rotation=[math.cos(math.pi/4), 0., 0., math.sin(math.pi/4)],
                 ego_dynamic_state=[float(i), 2., 3., 4.], driving_command=[0,1,0,0],
                 cams={"CAM_F0":{"data_path":f"{log}/{i}.jpg"}}) for i in range(n)]


def test_rear_axle_se2_has_no_y_flip_and_eight_future_samples(tmp_path):
    row = build_window(frames(), 3, "train", tmp_path)
    assert row["trajectory_gt"][0] == pytest.approx([1., 0., 0.], abs=1e-7)
    assert row["trajectory_gt"][-1] == pytest.approx([8., 0., 0.], abs=1e-7)
    assert row["ego"] == [3., 2., 3., 4.]
    assert row["command"] == 1
    assert row["history_times_s"] == [-1.5,-1.,-.5,0.]
    assert row["future_times_s"] == [.5,1.,1.5,2.,2.5,3.,3.5,4.]
    assert row["images"] == [str(tmp_path/f"log-a/{i}.jpg") for i in range(4)]
    assert "nav" not in row


def test_actual_timestamp_gap_and_native_onehot_are_validated(tmp_path):
    data=frames(); data[5]["timestamp"] += 100_000
    with pytest.raises(ValueError, match="timestamp"):
        build_window(data, 3, "train", tmp_path)
    data=frames(); data[3]["driving_command"]=[0,1,1,0]
    with pytest.raises(ValueError, match="one.hot"):
        build_window(data, 3, "train", tmp_path)


def test_heading_wrap_future_fields_cannot_change_inputs(tmp_path):
    data=frames(); before=build_window(data,3,"train",tmp_path)
    data[4]["ego2global_rotation"]=[math.cos(-math.pi/4),0,0,math.sin(-math.pi/4)]
    data[4]["ego_dynamic_state"]=[999]*4
    data[4]["driving_command"]=[1,0,0,0]
    after=build_window(data,3,"train",tmp_path)
    assert -math.pi <= after["trajectory_gt"][0][2] <= math.pi
    for key in ["images","ego","command","history_times_s"]:
        assert after[key] == before[key]


def test_logs_split_before_windows_and_stats_only_accept_train(tmp_path):
    split=split_logs(["c","b","a"], train_logs=2, val_logs=1)
    assert split == {"train":["a","b"],"val":["c"],"test":[]}
    train=build_window(frames(),3,"train",tmp_path)
    val=build_window(frames(log="log-b"),4,"val",tmp_path)
    stats=compute_ego_stats([train])
    assert stats["ego_mean"] == train["ego"]
    assert stats["source_split"] == "train"
    assert stats["ego_std"] == [1.,1.,1.,1.]
    with pytest.raises(ValueError, match="train"):
        compute_ego_stats([train,val])


def test_pose_matches_official_pyquaternion_intrinsic_yaw_and_rejects_nonfinite():
    frame=frames()[0]
    frame["ego2global_rotation"]=[.7,.2,.3,.6]
    w,x,y,z=frame["ego2global_rotation"];norm=math.sqrt(w*w+x*x+y*y+z*z)
    w,x,y,z=[v/norm for v in (w,x,y,z)]
    assert _pose(frame)[2] == pytest.approx(math.atan2(2*(w*z-x*y),1-2*(y*y+z*z)))
    frame["ego2global_translation"]=[float("nan"),0,0]
    with pytest.raises(ValueError,match="finite"):_pose(frame)


def test_left_is_positive_y_and_heading_crossing_wraps_to_small_angle(tmp_path):
    data=frames()
    for i in range(4,12):data[i]["ego2global_translation"]=[9.,3.,0.]
    row=build_window(data,3,"train",tmp_path)
    assert row["trajectory_gt"][0][:2]==pytest.approx([0.,1.],abs=1e-7)
    before=math.pi-.01;after=-math.pi+.02
    data[3]["ego2global_rotation"]=[math.cos(before/2),0,0,math.sin(before/2)]
    data[4]["ego2global_rotation"]=[math.cos(after/2),0,0,math.sin(after/2)]
    assert build_window(data,3,"train",tmp_path)["trajectory_gt"][0][2]==pytest.approx(.03)


def test_seed_is_consumed_in_log_split():
    names=[str(i) for i in range(12)]
    a=split_logs(names,8,2,seed=41)
    assert a==split_logs(names,8,2,seed=41)
    assert a!=split_logs(names,8,2,seed=42)


def test_dataset_rejects_malformed_shape_and_bad_stats(tmp_path):
    import json
    row=build_window(frames(),3,"train",tmp_path)
    stats=compute_ego_stats([row]);path=tmp_path/"manifest.jsonl"
    row["trajectory_gt"]=[[0.,0.,0.]]
    path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="8.*3"):NavsimDataset(path,stats)
    row=build_window(frames(),3,"train",tmp_path);path.write_text(json.dumps(row)+"\n")
    stats["ego_std"][0]=0.
    with pytest.raises(ValueError,match="positive"):NavsimDataset(path,stats)


def test_manifest_writer_rejects_leakage_before_writing(tmp_path):
    from fast_navsim.data import write_manifests
    row=build_window(frames(),3,"train",tmp_path)
    val=dict(row,split="val")
    with pytest.raises(ValueError,match="leakage"):
        write_manifests({"train":[row],"val":[val]},tmp_path/"bad")
    assert not (tmp_path/"bad/train.jsonl").exists()


@pytest.mark.parametrize("field,value",[("dataset","bad"),("split","private_test"),
 ("command",1.5),("command_onehot",[1,0,0,0]),("coordinate_frame","carla"),
 ("history_times_s",[0,1,2,3]),("future_times_s",[1]*8)])
def test_dataset_rejects_untrusted_manifest_contract(tmp_path,field,value):
    import json
    row=build_window(frames(),3,"train",tmp_path);stats=compute_ego_stats([row]);row[field]=value
    path=tmp_path/"manifest.jsonl";path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError):NavsimDataset(path,stats)


def test_available_windows_require_history_jpegs_and_no_future_jpegs():
    from fast_navsim.data import available_window_indices
    data=frames();available={f["cams"]["CAM_F0"]["data_path"] for f in data[:6]}
    assert available_window_indices(data,available)==[3,4,5]
    available.remove(data[2]["cams"]["CAM_F0"]["data_path"])
    assert available_window_indices(data,available)==[]


def test_local_smoke_builder_splits_logs_before_windows(tmp_path):
    from tools.prepare_navsim_smoke import build_from_retained
    records={}
    sensor=tmp_path/"sensor"
    for name in ["a","b","c"]:
        data=frames(log=name)
        records[name]={"frames":data,"metadata_sha256":"a"*64,"metadata_path":"fixture"}
        for frame in data[:6]:
            path=sensor/frame["cams"]["CAM_F0"]["data_path"]
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"fixture")
    report=build_from_retained(records,sensor,tmp_path/"out",seed=42,train_windows=2,val_windows=1)
    assert report["counts"]=={"train":4,"val":1,"test":0}
    assert report["provenance"]["split_seed"]==42
    assert (tmp_path/"out/deep_train.jsonl").read_text().count("\n")==1
    assert (tmp_path/"out/stats.json").exists()


def test_source_audit_contains_pinned_sources_and_detects_changed_bytes(tmp_path):
    import hashlib,json
    from tools.prepare_navsim_smoke import write_source_audit
    source=tmp_path/"source";source.mkdir();file=source/"official.py";file.write_bytes(b"official")
    digest=hashlib.sha256(b"official").hexdigest()
    (source/"source_identity.json").write_text(json.dumps({"navsim_revision":"a"*40,
       "files":[{"source_path":"official.py","sha256":digest,"url":"https://example.test/pin"}]}))
    report=write_source_audit(tmp_path/"source_audit.json",source,[],"b"*40,100)
    assert report["sources"][0]["path"]==str(file)
    assert report["sources"][0]["revision"]=="a"*40
    file.write_bytes(b"changed")
    with pytest.raises(ValueError,match="SHA256"):
        write_source_audit(tmp_path/"source_audit.json",source,[],"b"*40,100)


def test_dataset_checks_individual_timestamp_lengths_and_wrapped_heading(tmp_path):
    import json
    row=build_window(frames(),3,"train",tmp_path);stats=compute_ego_stats([row]);path=tmp_path/"rows"
    row["history_timestamps_us"].append(row["future_timestamps_us"].pop(0))
    path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="timestamp"):NavsimDataset(path,stats)
    row=build_window(frames(),3,"train",tmp_path);row["trajectory_gt"][0][2]=4.
    path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="heading"):NavsimDataset(path,stats)


def test_deep_row_selection_uses_current_speed_and_skips_stationary_rows():
    from fast_navsim.data import select_deep_row
    stationary={"split":"train","ego":[0.,0.,4.,5.],"trajectory_gt":[[100.,100.,0.]],"token":"stationary"}
    moving={"split":"train","ego":[.8,.6,0.,0.],"trajectory_gt":[[0.,0.,0.]],"token":"moving"}
    later={"split":"train","ego":[5.,0.,0.,0.],"token":"later"}
    assert select_deep_row([stationary,moving,later]) is moving
    moving["trajectory_gt"]=[[999.,999.,3.]]
    assert select_deep_row([stationary,moving,later]) is moving
    with pytest.raises(ValueError,match="speed"):
        select_deep_row([stationary])


def test_manifest_deep_selection_policy_records_moving_row(tmp_path):
    import json
    from fast_navsim.data import write_manifests
    data=frames();data[3]["ego_dynamic_state"]=[0.,0.,0.,0.]
    stationary=build_window(data,3,"train",tmp_path)
    moving=build_window(data,4,"train",tmp_path)
    val=build_window(frames(log="val-log"),3,"val",tmp_path)
    report=write_manifests({"train":[stationary,moving],"val":[val]},tmp_path/"out")
    assert json.loads((tmp_path/"out/deep_train.jsonl").read_text())["token"]==moving["token"]
    selection=report["deep_selection"]
    assert selection["row_index"]==1
    assert selection["log"]==moving["route"]
    assert selection["token"]==moving["token"]
    assert selection["min_speed_mps"]==1.
    assert selection["uses_future_targets"] is False
