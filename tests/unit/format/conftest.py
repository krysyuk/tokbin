from __future__ import annotations

import copy
from typing import Any

import pytest

_SHA = "a" * 64


def _meta() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tokbin_version": "0.1.0",
        "modality": "text",
        "packing": "stream",
        "dtype": "uint16",
        "item_shape": [],
        "byteorder": "little",
        "vocab_size": 50257,
        "tokenizer_id": "gpt2",
        "tokenizer_hash": "a3f2c81b9e04d5a7",
        "eos_id": 50256,
        "bos_id": None,
        "splits": {
            "train": {
                "created_at": "2026-09-26T10:22:31Z",
                "n_items": 150,
                "n_docs": 3,
                "n_skipped": 1,
                "has_split_docs": True,
                "shards": [
                    {"name": "train-00000.bin", "n_items": 100, "n_bytes": 200, "sha256": _SHA},
                    {"name": "train-00001.bin", "n_items": 50, "n_bytes": 100, "sha256": _SHA},
                ],
            },
            "valid": {
                "created_at": "2026-09-26T11:00:00Z",
                "n_items": 10,
                "n_docs": 1,
                "n_skipped": 0,
                "has_split_docs": False,
                "shards": [
                    {"name": "valid-00000.bin", "n_items": 10, "n_bytes": 20, "sha256": _SHA},
                ],
            },
        },
    }


def _checkpoint() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split": "train",
        "dtype": "uint16",
        "config_hash": "5" * 64,
        "tokenizer_hash": "a3f2c81b9e04d5a7",
        "n_input_consumed": 10,
        "pending_doc_items": 7,
        "last_input_id": "wiki.jsonl:9",
        "n_items": 100,
        "n_docs": 8,
        "n_skipped": 2,
        "has_split_docs": True,
        "closed_shards": [
            {"name": "train-00000.bin", "n_items": 100, "n_bytes": 200, "sha256": _SHA},
        ],
        "side_files": {"ids": 300, "ids_idx": 192, "offsets": 72, "skipped": 90},
        "updated_at": "2026-09-26T11:02:14Z",
    }


@pytest.fixture
def meta_dict() -> dict[str, Any]:
    """A valid meta.json document; tests mutate their own copy."""
    return copy.deepcopy(_meta())


@pytest.fixture
def checkpoint_dict() -> dict[str, Any]:
    """A valid checkpoint.json document; tests mutate their own copy."""
    return copy.deepcopy(_checkpoint())
