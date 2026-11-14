"""Optional file of query embeddings, so that retrieval can be repeated exactly.

`text-embedding-3-large` is not bit-reproducible: embedding the same string twice returns vectors
that differ in the last few digits, which can swap neighbors whose similarity scores are close.
With `LAMP_FROZEN_QUERY_EMBEDDINGS` set, query embeddings are read from and appended to a file, so
each distinct query string is embedded at most once and retrieves the same materials every time.
Only query vectors are cached; material vectors are stored in `MaterialEmbedding.embedding`. The
strings named in `pinned_texts` must be in the file; any other string is embedded, written to a
separate overlay file beside a read-only `base` file. With the variable unset,
`install_query_cache` returns the embeddings object it was given.

Two file roles:
  * a recording cache is plain ``.jsonl``, appended to, one entry per line;
  * the published artifact is ``.jsonl.gz``, written once by `finalize()`,
    atomically, and read-only afterwards.
Appending gzip members to a live file leaves a truncated final member if the
process dies, and a truncated member raises `EOFError` for the whole file.

Loading is fail-closed: every entry is re-derived and checked (key, dimension,
finiteness, conflicting duplicates), and anything inconsistent raises rather
than being skipped.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

#: Path to the cache. Unset means "no freezing".
ENV_PATH = "LAMP_FROZEN_QUERY_EMBEDDINGS"
#: ``auto`` (default) records misses; ``strict`` raises on one.
ENV_MODE = "LAMP_FROZEN_QUERY_EMBEDDINGS_MODE"

MODE_AUTO = "auto"
MODE_STRICT = "strict"


class FrozenQueryEmbeddingError(BaseException):
    """A pinned query was not in the frozen cache while running in strict mode.

    Derives from `BaseException` so that `except Exception` handlers on the
    agent path cannot swallow it, and the run stops. The wrapper that raises it
    is installed only when `LAMP_FROZEN_QUERY_EMBEDDINGS` is set.
    """


class FrozenCacheIntegrityError(RuntimeError):
    """The cache file is not a faithful record and must not be replayed."""


def _key(model: str, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_row(row: Any, where: str) -> Tuple[str, str, str, List[float]]:
    """Return (key, model, text, vector) or raise."""
    if not isinstance(row, dict):
        raise FrozenCacheIntegrityError(f"{where}: entry is not an object")
    for field in ("key", "model", "text", "vector"):
        if field not in row:
            raise FrozenCacheIntegrityError(f"{where}: missing {field!r}")
    key, model, text, vector = (
        row["key"], row["model"], row["text"], row["vector"],
    )
    if not isinstance(key, str) or not isinstance(model, str) or not isinstance(
        text, str
    ):
        raise FrozenCacheIntegrityError(f"{where}: key/model/text must be strings")
    # Re-derive the key: a mismatch means the vector belongs to another text.
    if _key(model, text) != key:
        raise FrozenCacheIntegrityError(
            f"{where}: key does not match sha256(model\\x00text)"
        )
    if not isinstance(vector, list) or not vector:
        raise FrozenCacheIntegrityError(f"{where}: vector must be a non-empty list")
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FrozenCacheIntegrityError(f"{where}: vector holds a non-number")
        if not math.isfinite(value):
            raise FrozenCacheIntegrityError(f"{where}: vector holds {value}")
    return key, model, text, [float(v) for v in vector]


def _check_collection(entries: List[Dict[str, Any]], where: str) -> List[Dict[str, Any]]:
    """Check dimensions and duplicates across entries for every model in the file.

    Returns the entries with identical duplicates removed; raises on mixed
    dimensions within a model or conflicting vectors for one key.
    """
    dims: Dict[str, int] = {}
    seen: Dict[str, List[float]] = {}
    unique: List[Dict[str, Any]] = []
    for entry in entries:
        model, key, vector = entry["model"], entry["key"], entry["vector"]
        dim = dims.setdefault(model, len(vector))
        if len(vector) != dim:
            raise FrozenCacheIntegrityError(
                f"{where}: vector dimension {len(vector)} != {dim} for "
                f"model {model!r}"
            )
        previous = seen.get(key)
        if previous is None:
            seen[key] = vector
            unique.append(entry)
        elif previous != vector:
            raise FrozenCacheIntegrityError(
                f"{where}: conflicting vectors for the same key {key[:16]}..."
            )
    return unique


def read_entries(path: str | Path) -> List[Dict[str, Any]]:
    """Parse and validate every entry in a cache file, raising on any inconsistency.

    A truncated final line in a plain ``.jsonl`` (left by an interrupted
    append) is dropped. Any other malformed line, and any problem in a ``.gz``
    artifact, raises. Identical duplicates are collapsed; conflicting ones raise.
    """
    path = Path(path)
    gzipped = path.suffix == ".gz"
    opener = (lambda: gzip.open(path, "rt", encoding="utf-8")
              if gzipped else path.open("r", encoding="utf-8"))
    try:
        with opener() as handle:
            raw = handle.read()
    except (EOFError, OSError, gzip.BadGzipFile) as exc:
        # A published artifact is written atomically and must be complete.
        raise FrozenCacheIntegrityError(f"{path}: unreadable ({exc})") from exc

    lines = raw.split("\n")
    trailing_newline = raw.endswith("\n")
    entries: List[Dict[str, Any]] = []
    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        is_final_fragment = (
            index == len(lines) - 1 and not trailing_newline and not gzipped
        )
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if is_final_fragment:
                logger.warning(
                    "%s: dropping a truncated final line (interrupted run)", path
                )
                continue
            raise FrozenCacheIntegrityError(
                f"{path} line {index + 1}: malformed JSON"
            )
        key, model, text, vector = _validate_row(row, f"{path} line {index + 1}")
        entries.append(
            {"key": key, "model": model, "text": text, "vector": vector}
        )
    return _check_collection(entries, str(path))


class FrozenQueryEmbeddings(Embeddings):
    """Wrap an embeddings object and serve `embed_query` from a frozen file.

    Subclasses ``Embeddings`` because FAISS checks
    ``isinstance(self.embedding_function, Embeddings)`` and otherwise calls the
    object directly. Other attributes are delegated to the wrapped object.

    The lock serializes threads within one process; concurrent processes
    sharing one recording cache are not supported.
    """

    def __init__(self, inner: Any, path: str | Path, *, model: str,
                 mode: str = MODE_AUTO, base: Optional[str | Path] = None,
                 pinned_texts: Optional[Any] = None) -> None:
        self._inner = inner
        self._path = Path(path)
        self._base = Path(base) if base else None
        self._model = model
        self._mode = mode
        # None means strict mode covers every query; an empty set covers none.
        self._pinned_texts = (
            None if pinned_texts is None else frozenset(pinned_texts)
        )
        self._pinned_keys = (
            None if self._pinned_texts is None
            else {_key(model, t) for t in self._pinned_texts}
        )
        self._lock = threading.Lock()
        self._vectors: Dict[str, List[float]] = {}
        self._texts: Dict[str, str] = {}
        self._dim: Optional[int] = None
        self._hits = 0
        self._missed_texts: List[str] = []
        self._ancillary_texts: List[str] = []
        self._strict_misses = 0
        if self._base is not None:
            self._load_file(self._base)
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return
        self._load_file(self._path)

    def _load_file(self, path: Path) -> None:
        if not path.exists():
            raise FrozenCacheIntegrityError(f"{path}: base artifact is missing")
        for entry in read_entries(path):
            if entry["model"] != self._model:
                continue
            self._remember(entry["key"], entry["text"], entry["vector"],
                           source=str(path))

    def _remember(self, key: str, text: str, vector: List[float], *,
                  source: str) -> None:
        if self._dim is None:
            self._dim = len(vector)
        elif len(vector) != self._dim:
            raise FrozenCacheIntegrityError(
                f"{source}: vector dimension {len(vector)} != {self._dim} "
                f"for model {self._model!r}"
            )
        previous = self._vectors.get(key)
        if previous is not None and previous != vector:
            # Two different vectors for one (model, text) cannot be replayed
            # deterministically.
            raise FrozenCacheIntegrityError(
                f"{source}: conflicting vectors for the same key {key[:16]}..."
            )
        self._vectors[key] = vector
        self._texts[key] = text

    def _append(self, key: str, text: str, vector: List[float]) -> None:
        if self._path.suffix == ".gz":
            raise FrozenCacheIntegrityError(
                f"refusing to append to the published artifact {self._path}; "
                "record to a plain .jsonl and publish it with finalize()"
            )
        # Repair an interrupted previous append so the new entry starts on its
        # own line.
        self._repair_tail()
        row = {"key": key, "model": self._model, "text": text, "vector": vector}
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _repair_tail(self) -> None:
        """Make the file safely appendable after an interrupted write.

        A complete final row without a newline is kept and newline-terminated
        (it is already loaded in memory). A partial final row is truncated,
        since terminating it would leave a malformed middle line that loading
        rejects. The file is scanned backwards in blocks, as entries are large.
        """
        if not self._path.exists():
            return
        block = 1 << 20
        with self._path.open("r+b") as handle:
            size = handle.seek(0, os.SEEK_END)
            if size == 0:
                return
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return
            position, start = size, 0
            tail = b""
            while position > 0:
                start = max(0, position - block)
                handle.seek(start)
                chunk = handle.read(position - start)
                index = chunk.rfind(b"\n")
                if index != -1:
                    tail = chunk[index + 1:] + tail
                    start = start + index + 1
                    break
                tail = chunk + tail
                position = start
            try:
                json.loads(tail.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                handle.truncate(start)  # partial row: cut it away
                return
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write("\n")  # complete row: just terminate it

    def embed_query(self, text: str) -> List[float]:
        key = _key(self._model, text)
        with self._lock:
            cached = self._vectors.get(key)
            if cached is not None:
                self._hits += 1
                return list(cached)
            # Strict mode applies to pinned texts only (all texts when none are
            # pinned). Strings the agent composes vary between runs and are
            # embedded live.
            is_pinned = (self._pinned_keys is None
                         or key in self._pinned_keys)
            if self._mode == MODE_STRICT and is_pinned:
                self._strict_misses += 1
                raise FrozenQueryEmbeddingError(
                    f"pinned query embedding not frozen (mode=strict): "
                    f"{text[:200]!r}"
                )
            vector = self._inner.embed_query(text)
            # Validate the provider response before it is written to the cache.
            _validate_row(
                {"key": key, "model": self._model, "text": text,
                 "vector": list(vector)},
                "provider response",
            )
            vector = [float(v) for v in vector]
            if self._dim is not None and len(vector) != self._dim:
                raise FrozenCacheIntegrityError(
                    f"provider response: vector dimension {len(vector)} != "
                    f"{self._dim} for model {self._model!r}"
                )
            # Persist before updating memory, so memory never holds a vector
            # the file lacks.
            self._append(key, text, vector)
            self._remember(key, text, vector, source="provider")
            self._missed_texts.append(text)
            if self._pinned_keys is not None and key not in self._pinned_keys:
                self._ancillary_texts.append(text)
            return list(vector)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Document embeddings (index builds) are delegated without freezing.
        return self._inner.embed_documents(texts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def finalize(self, dest: str | Path, *, force: bool = False) -> Dict[str, Any]:
        """Publish the recording cache as one complete gzip artifact; return its manifest.

        The file is written to a temporary path and renamed, so the destination
        is either absent or complete. An existing artifact is not overwritten
        unless `force` is set, since runs that share it must see the same vectors.
        """
        dest = Path(dest)
        if dest.exists() and not force:
            raise FrozenCacheIntegrityError(
                f"{dest} already exists; a published artifact is immutable. "
                "Pass force=True only to deliberately supersede it."
            )
        entries = read_entries(self._path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False))
                handle.write("\n")
        os.replace(tmp, dest)
        return manifest(dest, entries=entries)


def manifest(path: str | Path, *,
             entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Return a provenance record for a cache file (hashes, entry count, models, dims)."""
    path = Path(path)
    if entries is None:
        entries = read_entries(path)
    texts = sorted(e["text"] for e in entries)
    text_digest = hashlib.sha256()
    for text in texts:
        text_digest.update(text.encode("utf-8"))
        text_digest.update(b"\x00")
    models = sorted({e["model"] for e in entries})
    dims = sorted({len(e["vector"]) for e in entries})
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "entries": len(entries),
        "models": models,
        "vector_dims": dims,
        "query_set_sha256": text_digest.hexdigest(),
    }


def install_query_cache(embeddings: Any, *, model: str,
                        path: Optional[str] = None,
                        mode: Optional[str] = None) -> Any:
    """Return `embeddings`, wrapped iff a frozen-cache path is configured."""
    path = path if path is not None else os.environ.get(ENV_PATH)
    if not path:
        return embeddings
    mode = mode or os.environ.get(ENV_MODE, MODE_AUTO)
    if mode not in (MODE_AUTO, MODE_STRICT):
        raise ValueError(
            f"{ENV_MODE} must be {MODE_AUTO!r} or {MODE_STRICT!r}, got {mode!r}"
        )
    wrapper = FrozenQueryEmbeddings(embeddings, path, model=model, mode=mode)
    logger.info(
        "frozen query embeddings active: %s (%d entries, mode=%s)",
        path, len(wrapper._vectors), mode,
    )
    return wrapper
