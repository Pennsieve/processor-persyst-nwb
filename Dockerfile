# Pinned by digest, and not by tag. The `python:3.12` tag moves. This image is
# the first stage of a clinical pipeline, and its output must not change because
# a rebuild occurred on a different day. To update the digest, run:
#   docker pull python:3.12-slim-bookworm
#   docker image inspect python:3.12-slim-bookworm --format '{{index .RepoDigests 0}}'
FROM python@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

WORKDIR /app

# h5py normally installs from a manylinux wheel with HDF5 bundled; keep the
# system library available so a source fallback can still build.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libhdf5-dev \
	&& rm -rf /var/lib/apt/lists/*

# --require-hashes gives the lock file full control. The build fails if a version
# differs from the version that CI tested, or if an artifact has changed. To
# update the lock file, run `make lock`.
COPY processor/requirements.lock /app/processor/requirements.lock
RUN pip install --no-cache-dir --require-hashes \
	-r /app/processor/requirements.lock

COPY processor/ /app/processor

ENV PYTHONPATH="/app"

# The `import pynwb` statement builds a schema cache in XDG_CACHE_HOME before the
# converter starts. That path must be writable by any UID that the platform
# selects, and not only by appuser. The build therefore creates the cache and
# keeps it writable for all users, and HOME points to a writable directory.
# Without these two settings, the image runs only as root.
ENV HOME=/tmp \
	XDG_CACHE_HOME=/app/.cache
RUN mkdir -p /app/.cache \
	&& python3.12 -c "import pynwb" \
	&& chmod -R 0777 /app/.cache

# The converter does not need root. OUTPUT_DIR is a bind mount, and this UID must
# be able to write to it. For a local run, docker-compose.yml gives the UID of
# the host user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

CMD ["python3.12", "-m", "processor.main"]
