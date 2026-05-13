"""AESTETIK: AutoEncoder for Spatial Transcriptomics Expression with
Topology and Image Knowledge."""
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from aestetik.AESTETIK import AESTETIK

try:
    __version__ = _version("aestetik")
except PackageNotFoundError:  # not installed (running from source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["AESTETIK", "__version__"]
