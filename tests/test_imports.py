"""Smoke test that every aestetik subpackage installs and imports.

Regression test for issue #6: five subpackages were missing
``__init__.py`` files and were silently dropped by ``find_packages``.
"""
import importlib

import pytest

REQUIRED_MODULES = [
    "aestetik",
    "aestetik.AESTETIK",
    "aestetik.dataloader",
    "aestetik.callbacks",
    "aestetik.callbacks.callbacks",
    "aestetik.data_modules",
    "aestetik.data_modules.data_module",
    "aestetik.metrics",
    "aestetik.metrics.loss_function",
    "aestetik.models",
    "aestetik.models.model",
    "aestetik.modules",
    "aestetik.modules.aestetik_module",
    "aestetik.utils",
    "aestetik.utils._pyvips_dtype",
    "aestetik.utils.utils_clustering",
    "aestetik.utils.utils_data",
    "aestetik.utils.utils_grid",
    "aestetik.utils.utils_morphology",
    "aestetik.utils.utils_transcriptomics",
]

# utils_visualization is a special case: it imports squidpy / matplotlib
# but is otherwise self-contained. Keep it in its own test so a heavy
# import failure (e.g. squidpy not wheel-built for a new python) shows
# up clearly.
def test_utils_visualization_importable() -> None:
    pytest.importorskip("squidpy")
    importlib.import_module("aestetik.utils.utils_visualization")


@pytest.mark.parametrize("name", REQUIRED_MODULES)
def test_module_importable(name: str) -> None:
    importlib.import_module(name)


def test_public_api() -> None:
    from aestetik import AESTETIK
    assert AESTETIK is not None
