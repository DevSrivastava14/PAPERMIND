"""Process-local cache for complete document ingestion and retrieval bundles."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import os
import threading
from typing import Callable, Generic, TypeVar


T = TypeVar("T")
MAX_CACHED_BUNDLES = 4


@dataclass(frozen=True)
class DocumentFingerprint:
    size: int
    modified_ns: int


@dataclass(frozen=True)
class DocumentBundle:
    document: object
    pipeline: object


_cache: "OrderedDict[tuple[str, str, DocumentFingerprint], DocumentBundle]" = OrderedDict()
_locks: dict[tuple[str, str], threading.Lock] = {}
_lock_users: dict[tuple[str, str], int] = {}
_cache_lock = threading.RLock()


def _fingerprint(path: Path) -> DocumentFingerprint:
    stat = path.stat()
    return DocumentFingerprint(size=stat.st_size, modified_ns=stat.st_mtime_ns)


def get_or_build_bundle(
    path: Path,
    document_id: str,
    builder: Callable[[], DocumentBundle],
) -> DocumentBundle:
    """Return a complete bundle, building it once for concurrent callers."""
    resolved = path.resolve()
    fingerprint = _fingerprint(resolved)
    identity = (document_id, str(resolved))
    key = (identity[0], identity[1], fingerprint)

    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
        build_lock = _locks.setdefault(identity, threading.Lock())
        _lock_users[identity] = _lock_users.get(identity, 0) + 1

    try:
        with build_lock:
            fingerprint = _fingerprint(resolved)
            key = (identity[0], identity[1], fingerprint)
            with _cache_lock:
                cached = _cache.get(key)
                if cached is not None:
                    _cache.move_to_end(key)
                    return cached

            bundle = builder()
            if bundle.document is None or bundle.pipeline is None:
                raise ValueError("Document cache builder returned an incomplete bundle")

            if _fingerprint(resolved) != fingerprint:
                return bundle

            with _cache_lock:
                stale_keys = [
                    existing_key
                    for existing_key in _cache
                    if existing_key[0] == identity[0] and existing_key[1] == identity[1]
                ]
                for stale_key in stale_keys:
                    _cache.pop(stale_key, None)
                _cache[key] = bundle
                _cache.move_to_end(key)
                while len(_cache) > int(os.getenv("PAPER_MIND_CACHE_SIZE", MAX_CACHED_BUNDLES)):
                    _cache.popitem(last=False)
            return bundle
    finally:
        with _cache_lock:
            _lock_users[identity] -= 1
            if _lock_users[identity] == 0:
                _lock_users.pop(identity, None)
                _locks.pop(identity, None)


def clear_document_cache() -> None:
    """Clear cached bundles; intended for tests and controlled application teardown."""
    with _cache_lock:
        _cache.clear()
        _locks.clear()
        _lock_users.clear()
