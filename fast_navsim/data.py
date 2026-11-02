"""NAVSIM 2 Hz rear-axle windows, with original front camera JPEGs.

Public pure-Python builder API: build_window(frames, current_index, split,
sensor_root), split_logs(names, train_logs, val_logs), compute_ego_stats(rows),
and write_manifests(rows_by_split, destination, provenance). Torch is imported
only by NavsimDataset.__getitem__, keeping acquisition independent of CUDA.
"""
import json
import math
import random
from pathlib import Path

HISTORY_TIMES = [-1.5, -1.0, -0.5, 0.0]
FUTURE_TIMES = [0.5 * i for i in range(1, 9)]


def _pose(frame):
    q = [float(v) for v in frame["ego2global_rotation"]]
    norm = math.sqrt(sum(v*v for v in q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid ego quaternion")
    w,x,y,z = [v/norm for v in q]
    # Match Quaternion(...).yaw_pitch_roll[0], the official NAVSIM path.
    heading = math.atan2(2*(w*z-x*y), 1-2*(y*y+z*z))
    translation = frame["ego2global_translation"]
    if not all(math.isfinite(float(v)) for v in translation[:2]):
        raise ValueError("Ego XY translation must be finite")
    return float(translation[0]), float(translation[1]), heading


def build_window(frames, current_index, split, sensor_root, source=None):
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    if current_index < 3 or current_index + 8 >= len(frames):
        raise ValueError("Window needs four history and eight future frames")
    selected = frames[current_index-3:current_index+9]
    timestamps = [int(frame["timestamp"]) for frame in selected]
    if any(abs((b-a)-500_000) > 1000 for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError("NAVSIM timestamp gaps must be 500000 microseconds (+/-1000)")
    frame = frames[current_index]
    command = [float(x) for x in frame["driving_command"]]
    if len(command) != 4 or sum(command) != 1 or any(x not in (0,1) for x in command):
        raise ValueError("driving_command must be a four-way one-hot vector")
    ego = [float(v) for v in frame["ego_dynamic_state"]]
    if len(ego) != 4 or not all(math.isfinite(v) for v in ego):
        raise ValueError("ego_dynamic_state must have vx,vy,ax,ay")
    origin_x,origin_y,origin_h = _pose(frame)
    cos_h,sin_h = math.cos(origin_h),math.sin(origin_h)
    trajectory = []
    for future in frames[current_index+1:current_index+9]:
        x,y,h = _pose(future)
        dx,dy = x-origin_x,y-origin_y
        trajectory.append([cos_h*dx+sin_h*dy, -sin_h*dx+cos_h*dy,
                           math.atan2(math.sin(h-origin_h),math.cos(h-origin_h))])
    root = Path(sensor_root).resolve()
    image_paths = []
    for item in frames[current_index-3:current_index+1]:
        relative = Path(item["cams"]["CAM_F0"]["data_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Camera path must be relative to sensor root")
        image_paths.append(str(root / relative))
    log_name = frame["log_name"]
    if any(item.get("log_name", log_name) != log_name for item in selected):
        raise ValueError("Window crosses logs")
    row = {"dataset":"navsim", "split":split, "route":log_name,
           "scene":frame.get("scene_token", frame.get("scene_name",log_name)),
           "frame":current_index, "token":frame["token"], "images":image_paths,
           "ego":ego, "command":command.index(1), "command_onehot":command,
           "trajectory_gt":trajectory, "history_times_s":HISTORY_TIMES.copy(),
           "future_times_s":FUTURE_TIMES.copy(), "timestamp_us":timestamps[3],
           "history_timestamps_us":timestamps[:4], "future_timestamps_us":timestamps[4:],
           "source":source or {}, "coordinate_frame":"current_rear_axle_se2",
           "timestamp_unit":"microseconds"}
    return row


def split_logs(names, train_logs=2, val_logs=1, seed=None):
    names = sorted(set(names))
    if train_logs < 1 or val_logs < 1 or len(names) < train_logs+val_logs:
        raise ValueError("Need independent train and val logs")
    if seed is not None:
        random.Random(seed).shuffle(names)
    return {"train":names[:train_logs], "val":names[train_logs:train_logs+val_logs],
            "test":names[train_logs+val_logs:]}


def available_window_indices(frames, available_image_paths):
    """Find full four-image histories; future sensor images are never required."""
    available=set(available_image_paths)
    indices={i for i,frame in enumerate(frames)
             if frame["cams"]["CAM_F0"]["data_path"] in available}
    return [i for i in range(3,len(frames)-8)
            if all(i-d in indices for d in range(4))]


def compute_ego_stats(rows):
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("ego statistics require nonempty train rows only")
    mean = [sum(row["ego"][i] for row in rows)/len(rows) for i in range(4)]
    std = [math.sqrt(sum((row["ego"][i]-mean[i])**2 for row in rows)/len(rows))
           for i in range(4)]
    std = [value if value>=1e-6 else 1.0 for value in std]
    return {"ego_mean":mean, "ego_std":std, "source_split":"train", "count":len(rows),
            "features":["vx","vy","ax","ay"],
            "normalization_policy":"population_std; unit std for constant dimensions",
            "unitstd_for_constant":True}


def validate_row(row):
    if row.get("dataset")!="navsim" or row.get("split") not in ("train","val","test"):
        raise ValueError("Invalid NAVSIM dataset or split")
    trajectory=row["trajectory_gt"]
    if len(trajectory)!=8 or any(len(pose)!=3 for pose in trajectory):
        raise ValueError("trajectory_gt must have shape [8,3]")
    if len(row["images"])!=4 or len(row["ego"])!=4:
        raise ValueError("NAVSIM requires four images and four ego features")
    if any(not Path(path).is_absolute() for path in row["images"]):
        raise ValueError("Camera paths must be absolute")
    if not all(math.isfinite(float(v)) for v in row["ego"]+[v for pose in trajectory for v in pose]):
        raise ValueError("Nonfinite NAVSIM input or target")
    if any(abs(pose[2])>math.pi for pose in trajectory):
        raise ValueError("Target heading must be wrapped to [-pi,pi]")
    command=row["command"]
    if not isinstance(command,int) or isinstance(command,bool) or command not in range(4):
        raise ValueError("NAVSIM command index must be integer 0..3")
    if "command_onehot" in row:
        vector=row["command_onehot"]
        if len(vector)!=4 or vector!=[float(i==command) for i in range(4)]:
            raise ValueError("Command index must agree with native one-hot")
    if row.get("coordinate_frame")!="current_rear_axle_se2":
        raise ValueError("NAVSIM targets require current rear axle SE2")
    if row.get("history_times_s")!=HISTORY_TIMES or row.get("future_times_s")!=FUTURE_TIMES:
        raise ValueError("NAVSIM timing must match 2Hz history and future")
    if row.get("timestamp_unit")!="microseconds":
        raise ValueError("Actual timestamps must be microseconds")
    if len(row["history_timestamps_us"])!=4 or len(row["future_timestamps_us"])!=8:
        raise ValueError("Need four history and eight future timestamps")
    timestamps=row["history_timestamps_us"]+row["future_timestamps_us"]
    if len(timestamps)!=12 or any(not isinstance(v,int) for v in timestamps):
        raise ValueError("Need twelve integer actual timestamps")
    if row["timestamp_us"]!=timestamps[3] or any(abs(b-a-500_000)>1000 for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError("Actual timestamp gaps disagree with 2Hz window")


def select_deep_row(rows, min_speed_mps=1.0):
    """Select the first moving TRAIN row using only current raw ego velocity."""
    if not math.isfinite(min_speed_mps) or min_speed_mps<=0:
        raise ValueError("Deep selection minimum speed must be finite and positive")
    if any(row.get("split")!="train" for row in rows):
        raise ValueError("Deep selection requires TRAIN rows only")
    for row in rows:
        speed=math.hypot(float(row["ego"][0]),float(row["ego"][1]))
        if math.isfinite(speed) and speed>=min_speed_mps:
            return row
    raise ValueError(f"No TRAIN row meets current speed >= {min_speed_mps} m/s")


def write_manifests(rows_by_split, destination, provenance=None):
    if not rows_by_split.get("train") or not rows_by_split.get("val"):
        raise ValueError("Need nonempty train and independent val rows")
    splits = {split:sorted({row["route"] for row in rows_by_split.get(split,[])})
              for split in ("train","val","test")}
    flat = [name for values in splits.values() for name in values]
    if len(flat) != len(set(flat)):
        raise ValueError("Log leakage between splits")
    seen_tokens={}
    for split in splits:
        for row in rows_by_split.get(split,[]):
            validate_row(row)
            if row["split"]!=split:
                raise ValueError("Row split label disagrees with destination")
            if row["token"] in seen_tokens:
                raise ValueError("Token leakage or duplicate token")
            seen_tokens[row["token"]]=split
    stats=compute_ego_stats(rows_by_split["train"])
    train=rows_by_split["train"]
    deep_row=select_deep_row(train)
    deep_index=next(index for index,row in enumerate(train) if row is deep_row)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train","val","test"):
        rows = rows_by_split.get(split, [])
        (destination/f"{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    (destination/"deep_train.jsonl").write_text(json.dumps(deep_row)+"\n")
    (destination/"stats.json").write_text(json.dumps(stats,indent=2)+"\n")
    (destination/"splits.json").write_text(json.dumps(splits,indent=2)+"\n")
    report = {"counts":{key:len(rows_by_split.get(key,[])) for key in splits},
              "splits":splits,"provenance":provenance or {},
              "protocol":"bounded mini smoke; not an official navtrain/navtest score",
              "deep_selection":{"policy":"first TRAIN row with current hypot(vx,vy) >= min_speed_mps; inputs only",
                                "min_speed_mps":1.0,"uses_future_targets":False,
                                "row_index":deep_index,"index":deep_index,
                                "log":deep_row["route"],"token":deep_row["token"],
                                "current_speed_mps":math.hypot(*deep_row["ego"][:2])}}
    (destination/"build_report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


class NavsimDataset:
    def __init__(self, manifest, stats):
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        self.stats = json.loads(Path(stats).read_text()) if isinstance(stats,(str,Path)) else stats
        if self.stats.get("source_split") != "train":
            raise ValueError("Dataset requires TRAIN-only ego statistics")
        mean,std=self.stats["ego_mean"],self.stats["ego_std"]
        if len(mean)!=4 or len(std)!=4 or not all(math.isfinite(v) for v in mean+std) or any(v<=0 for v in std):
            raise ValueError("Ego statistics require four finite means and positive finite standard deviations")
        for row in self.rows:
            validate_row(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        row = self.rows[index]
        images = []
        for path in row["images"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        ego = [(x-m)/s for x,m,s in zip(row["ego"],self.stats["ego_mean"],self.stats["ego_std"])]
        return {"images":images, "ego":torch.tensor(ego,dtype=torch.float32),
                "command":torch.tensor(row["command"],dtype=torch.long),
                "trajectory_gt":torch.tensor(row["trajectory_gt"],dtype=torch.float32),
                "metadata":dict(row)}
