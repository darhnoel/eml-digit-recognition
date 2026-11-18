"""Load MNIST from the IDX files in data/mnist/ (downloaded from the public mirror
https://storage.googleapis.com/cvdf-datasets/mnist/). Parsed as plain bytes with
NumPy; nothing in the files is executed.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "mnist"
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
SIZES = {  # bytes of the official gzip files, checked after download
    "train-images-idx3-ubyte.gz": 9912422,
    "train-labels-idx1-ubyte.gz": 28881,
    "t10k-images-idx3-ubyte.gz": 1648877,
    "t10k-labels-idx1-ubyte.gz": 4542,
}


def download(data_dir: Path = DATA) -> None:
    """Download the four MNIST files into data/mnist/ (skips files already present)."""
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    for name, size in SIZES.items():
        path = data_dir / name
        if path.exists() and path.stat().st_size == size:
            continue
        urllib.request.urlretrieve(MIRROR + name, path)
        if path.stat().st_size != size:
            raise RuntimeError(f"{name}: unexpected size {path.stat().st_size}")
        print(f"downloaded {name}")


def parse_idx(raw: bytes) -> np.ndarray:
    """Parse an IDX file (unsigned bytes only, as MNIST uses)."""
    if raw[0] != 0 or raw[1] != 0 or raw[2] != 0x08:
        raise ValueError("not an unsigned-byte IDX file")
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != int(np.prod(dims)):
        raise ValueError("IDX size does not match its header")
    return data.reshape(dims)


def load(split: str, data_dir: Path = DATA) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): X float64 (n, 784) with raw pixel values 0..255, y int64."""
    img_file, lab_file = FILES[split]
    X = parse_idx(gzip.decompress((data_dir / img_file).read_bytes()))
    y = parse_idx(gzip.decompress((data_dir / lab_file).read_bytes()))
    return X.reshape(len(X), -1).astype(np.float64), y.astype(np.int64)


if __name__ == "__main__":
    download()
