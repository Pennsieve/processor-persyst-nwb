"""HDF5 chunk sizing for the sample and timestamp datasets."""

from processor.constants import MAX_CHUNK_SAMPLES, TARGET_CHUNK_BYTES


def chunk_samples(
    n_samples: int,
    n_channels: int,
    itemsize: int,
    target_bytes: int = TARGET_CHUNK_BYTES,
) -> int:
    """Choose how many samples one chunk should span.

    Sized by bytes rather than by a fixed sample count, so a 4-channel int16 file
    and a 256-channel int16 file both land near ``target_bytes`` instead of
    differing 64-fold.

    The result never exceeds ``n_samples``, because ``GenericDataChunkIterator``
    asserts each chunk axis fits within the dataset and a short recording would
    otherwise fail outright.

    The result never exceeds ``MAX_CHUNK_SAMPLES``, which bounds how much a
    reader must decompress to reach one sample when a recording is narrow enough
    that the byte budget alone would allow a chunk spanning millions of samples.

    Raise ValueError if any argument is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples!r}")
    if n_channels <= 0:
        raise ValueError(f"n_channels must be positive, got {n_channels!r}")
    if itemsize <= 0:
        raise ValueError(f"itemsize must be positive, got {itemsize!r}")
    if target_bytes <= 0:
        raise ValueError(f"target_bytes must be positive, got {target_bytes!r}")

    by_budget = target_bytes // (n_channels * itemsize)
    ceiling = min(n_samples, MAX_CHUNK_SAMPLES)
    return max(1, min(by_budget, ceiling))
