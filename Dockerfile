FROM python:3.12

WORKDIR /app

# h5py normally installs from a manylinux wheel with HDF5 bundled; keep the
# system library available so a source fallback can still build.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libhdf5-dev \
	&& rm -rf /var/lib/apt/lists/*

COPY processor/requirements.txt /app/processor/requirements.txt
RUN pip install --no-cache-dir -r /app/processor/requirements.txt

COPY processor/ /app/processor

ENV PYTHONPATH="/app"

CMD ["python3.12", "-m", "processor.main"]
