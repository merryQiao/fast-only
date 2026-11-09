"""Frame-policy boundaries: one real current image or the unchanged four history."""
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from PIL import Image

from fast_navsim.collate import HISTORY_TIMES, NavsimCollator, temporal_content
from fast_cluster.data import PreparedData, prepare_smoke_data
from slowfast_v11.model import FastWithSlowKV, SlowPrefill, assemble
from tests_cluster.test_data import smoke_fixture
from tests_slowfast.conftest import language_factory, native_input


def sample(frames=4):
    return {"images": [Image.new("RGB", (65, 35), (30 + index, 70, 100)) for index in range(frames)],
            "ego": torch.ones(4), "command": torch.tensor(1), "trajectory_gt": torch.zeros(8, 3)}


def test_original_four_frame_prompt_is_byte_identical():
    frames = sample()["images"]
    expected = []
    for time, image in zip((-1.5, -1., -.5, 0.), frames):
        expected.extend([{"type": "text", "text": f"Front camera t{time:+.1f}s: "},
                         {"type": "image", "image": image}])
    expected.append({"type": "text", "text": "Plan eight future ego poses (x, y, heading) at 0.5 second intervals over 4 seconds."})
    assert temporal_content(frames) == expected
    assert temporal_content(frames, history_times_s=HISTORY_TIMES) == expected


def test_single_frame_prompt_contains_exactly_one_current_image():
    frame = sample(1)["images"][0]
    content = temporal_content([frame], history_times_s=(0.,))
    assert content[0] == {"type": "text", "text": "Front camera t+0.0s: "}
    assert [item["image"] for item in content if item["type"] == "image"] == [frame]
    assert len(content) == 3
    with pytest.raises(ValueError):
        temporal_content([frame])
    with pytest.raises(ValueError):
        temporal_content([frame] * 4, history_times_s=(0.,))


@pytest.mark.parametrize("times", [(False,), (-.5, 0.), (1.,), (-1., -.5, 0.)])
def test_unsupported_frame_time_policies_fail_before_processor_loading(times):
    with pytest.raises(ValueError):
        NavsimCollator("unused", history_times_s=times)


@pytest.fixture(scope="module")
def collators():
    return {"default": NavsimCollator("models/Qwen3.5-0.8B"),
            4: NavsimCollator("models/Qwen3.5-0.8B", history_times_s=HISTORY_TIMES),
            1: NavsimCollator("models/Qwen3.5-0.8B", history_times_s=(0.,))}


def test_native_collator_keeps_four_default_and_one_actual_image(collators):
    old = sample()
    default, explicit = collators["default"]([old]), collators[4]([old])
    for key in default["native"]:
        assert torch.equal(default["native"][key], explicit["native"][key]), key
    single = collators[1]([sample(1)])
    assert single["native"]["image_grid_thw"].tolist() == [[1, 4, 6]]
    assert default["native"]["image_grid_thw"].tolist() == [[1, 4, 6]] * 4
    assert len(single["image_audit"]) == 1
    assert single["image_audit"][0]["source_size_wh"] == [65, 35]
    assert single["image_audit"][0]["resized"] is False
    with pytest.raises(ValueError):
        collators[1]([sample(4)])
    with pytest.raises(ValueError):
        collators[4]([sample(1)])


def one_native(native):
    ids = torch.tensor([[2, 60, 61, 63, 6]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == 61).long(), "image_grid_thw": native["image_grid_thw"][:1],
            "pixel_values": native["pixel_values"][:4]}


def test_one_frame_fast_fuses_four_frame_slow_and_reports_native_layout(language_factory, native_input):
    kw = dict(lora_rank=2, lora_alpha=4, lora_dropout=0., position_offset=128)
    fast = FastWithSlowKV(language_factory(), image_frames=1, **kw).eval()
    slow = SlowPrefill(language_factory(64), **kw).eval()
    ego, command = torch.ones(1, 4), torch.tensor([1])
    current = one_native(native_input)
    with torch.no_grad():
        candidate = fast(current, ego, command)
        source = slow(native_input, ego, command, feedback=candidate)
        result = fast(current, ego, command, snapshot=source.snapshot,
                      metadata=torch.tensor([[0., 1., 0., 0., 1., 1.]]))
    assert result.shape == (1, 8, 3) and torch.isfinite(result).all()
    assert len(source.snapshot.layers) == 6
    assert fast.last_layout["image_frames"] == 1
    assert fast.last_layout["image_grid_thw"] == [[1, 2, 2]]
    assert fast.last_layout["image_tokens_per_frame"] == [1]
    assert fast.last_layout["image_tokens_total"] == 1
    assert slow.image_frames == 4
    with pytest.raises(ValueError, match="image"):
        fast(native_input, ego, command)
    with pytest.raises(ValueError, match="image"):
        slow(current, ego, command)


@pytest.mark.parametrize("count", [0, 2, 3, 5, True, 1.0])
def test_fast_rejects_unsupported_image_count_before_initializing(count):
    with pytest.raises(ValueError, match="image_frames"):
        FastWithSlowKV(object(), image_frames=count)


def test_assemble_legacy_planner_defaults_to_four(language_factory, native_input):
    backbone = language_factory().eval()
    legacy = SimpleNamespace(backbone=backbone)
    explicit = SimpleNamespace(backbone=backbone, image_frames=4)
    ego, command, suffix = torch.ones(1, 4), torch.tensor([0]), torch.zeros(1, 11, 32)
    with torch.no_grad():
        old = assemble(legacy, native_input, ego, command, suffix)
        four = assemble(explicit, native_input, ego, command, suffix)
    assert all(torch.equal(a, b) for a, b in zip(old, four))
    with pytest.raises(ValueError, match="image"):
        assemble(legacy, one_native(native_input), ego, command, suffix)


def test_single_dataset_reads_only_latest_before_hash_and_decode(tmp_path, monkeypatch):
    root, train_path, val_path, rows, _ = smoke_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    prepare_smoke_data(train_path, val_path, root, bundle)
    prepared = PreparedData(bundle, root, allow_smoke=True)
    single = prepared.dataset("train", image_frames=1)
    opened = []
    original = Path.read_bytes
    def track(path):
        if path.suffix == ".jpg":
            opened.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", track)
    item = single[(0, False)]
    assert len(item["images"]) == len(item["image_paths"]) == 1
    assert item["valid"] is False
    assert opened == [Path(rows[0]["images"][-1])]
    assert len(single.row(0)["images"]) == 4
    assert len(prepared.dataset("train")[0]["images"]) == 4
    with pytest.raises(ValueError, match="image_frames"):
        prepared.dataset("train", image_frames=2)


def test_past_and_future_jpeg_substitution_cannot_change_single_native_input(tmp_path, collators):
    root, train_path, val_path, rows, _ = smoke_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    prepare_smoke_data(train_path, val_path, root, bundle)
    prepared = PreparedData(bundle, root, allow_smoke=True)
    single = prepared.dataset("train", image_frames=1)
    def native():
        item = single[0]
        item.update(ego=torch.tensor(item["ego"]), command=torch.tensor(item["command"]),
                    trajectory_gt=torch.tensor(item["trajectory_gt"]))
        return collators[1]([item])["native"]
    before = native()
    paths = rows[0]["images"][:3]
    paths.append(str(Path(rows[0]["images"][-1]).with_name("12.jpg")))
    for path in paths:
        Image.new("RGB", (17, 11), (200, 20, 10)).save(path)
    after = native()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    with pytest.raises(ValueError, match="image.*identity"):
        prepared.dataset("train", image_frames=4)[0]
