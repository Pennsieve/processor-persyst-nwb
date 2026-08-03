"""HDF5 chunk sizing for the sample and timestamp datasets."""

from processor.constants import MAX_CHUNK_SAMPLES, TARGET_CHUNK_BYTES


def chunk_samples(
    n_samples: int,
    n_channels: int,
    itemsize: int,
    target_bytes: int = TARGET_CHUNK_BYTES,
) -> int:
    """Choose how many samples one chunk should span.

    Sizing by bytes rather than by a fixed sample count keeps a 4-channel int16
    file and a 256-channel int16 file both near ``target_bytes``, instead of 64
    times apart.

    Two ceilings apply. ``n_samples`` is one: ``GenericDataChunkIterator``
    asserts that each chunk axis fits within the dataset, so a chunk longer than
    a short recording fails outright. ``MAX_CHUNK_SAMPLES`` is the other. It
    bounds how much a reader must decompress to reach one sample, which the byte
    budget alone does not do on a narrow recording.

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
