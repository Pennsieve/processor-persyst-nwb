import pytest

from processor.constants import MAX_CHUNK_SAMPLES
from processor.sizing import chunk_samples


def test_chunk_stays_within_byte_budget():
    n = chunk_samples(1_000_000, 35, 4, target_bytes=4 * 2**20)
    assert n * 35 * 4 <= 4 * 2**20


@pytest.mark.parametrize(
    ("n_channels", "itemsize"),
    [(4, 2), (35, 4), (64, 2), (256, 2), (512, 4)],
)
def test_chunk_never_exceeds_budget_for_any_width(n_channels, itemsize):
    target = 4 * 2**20
    n = chunk_samples(10_000_000, n_channels, itemsize, target_bytes=target)
    assert n * n_channels * itemsize <= target


def test_chunk_capped_for_narrow_recordings():
    # Bounds how much must be decompressed to reach a single sample.
    n = chunk_samples(10_000_000, 1, 2, target_bytes=2**30)
    assert n == MAX_CHUNK_SAMPLES


def test_chunk_clamped_to_short_recording():
    # GenericDataChunkIterator asserts each chunk axis fits the dataset, so the
    # 7587-sample NeuroPace recording must not get a 32768-sample chunk.
    assert chunk_samples(7587, 4, 2) == 7587


def test_chunk_at_least_one_for_very_wide_data():
    assert chunk_samples(1000, 100_000, 8, target_bytes=1024) == 1


@pytest.mark.parametrize(
    ("n_samples", "n_channels", "itemsize", "target"),
    [
        (0, 4, 2, 1024),
        (-1, 4, 2, 1024),
        (10, 0, 2, 1024),
        (10, 4, 0, 1024),
        (10, 4, 2, 0),
    ],
)
def test_non_positive_arguments_raise(n_samples, n_channels, itemsize, target):
    with pytest.raises(ValueError, match="must be positive"):
        chunk_samples(n_samples, n_channels, itemsize, target_bytes=target)
