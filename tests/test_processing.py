import numpy as np
import pytest
import torch
from PIL import Image

from fast_only.common import NativeCollator, prepare_original_image


def test_original_image_padding_preserves_every_source_pixel():
    pixels = np.arange(35*65*3, dtype=np.uint8).reshape(35, 65, 3)
    image, audit = prepare_original_image(Image.fromarray(pixels), 32, [0, 0, 0])
    assert image.size == (96, 64)
    np.testing.assert_array_equal(np.asarray(image)[:35, :65], pixels)
    assert np.asarray(image)[35:, :].max() == 0
    assert audit['source_size_wh'] == [65, 35]
    assert audit['processed_size_wh'] == [96, 64]
    assert audit['resized'] is False
    assert audit['padding_ltrb'] == [0, 0, 31, 29]


def test_aligned_original_image_does_not_change():
    image = Image.new('RGB', (64, 96), (17, 31, 53))
    padded, audit = prepare_original_image(image, 32, [0, 0, 0])
    assert padded.tobytes() == image.tobytes()
    assert audit['padding_ltrb'] == [0, 0, 0, 0]
    with pytest.raises(ValueError):
        prepare_original_image(image, 0, [0, 0, 0])


def test_real_processor_keeps_padded_original_resolution_and_seeded_augmentation():
    collator = NativeCollator('models/Qwen3.5-0.8B', policy='original_pad',
                             min_pixels=None, max_pixels=None, pad_multiple=32,
                             pad_fill=[0, 0, 0], prompt_template='temporal_task_v1', brightness=.1)
    sample = {'images': [Image.new('RGB', (65, 35), (90, 120, 150)) for _ in range(4)],
              'ego': torch.ones(3), 'nav': torch.ones(2), 'command': torch.tensor(4),
              'trajectory_gt': torch.zeros(6, 2)}
    first, second = collator([sample], seed=91), collator([sample], seed=91)
    assert first['native']['image_grid_thw'].tolist() == [[1, 4, 6]] * 4
    assert torch.equal(first['native']['pixel_values'], second['native']['pixel_values'])
    assert first['image_audit'][0]['source_size_wh'] == [65, 35]
    assert first['image_audit'][0]['processed_size_wh'] == [96, 64]
    assert first['image_audit'][0]['resized'] is False
