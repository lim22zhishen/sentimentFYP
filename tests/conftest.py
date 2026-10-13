"""Shared pytest setup."""

import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``torch`` when PyTorch isn't installed.

    Most of the suite runs without the multi-GB torch install; only diarization
    (which hands pyannote a torch tensor) needs it.
    """
    if importlib.util.find_spec("torch") is not None:
        return
    skip = pytest.mark.skip(reason="needs PyTorch (diarization builds a torch tensor)")
    for item in items:
        if "torch" in item.keywords:
            item.add_marker(skip)
