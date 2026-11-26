"""Search vectors saved on disk, so a big file is indexed once.

One file per case and embedding model:  data/index/<case_id>/<model>.vec
Format: b"IMEV1" + uint32 dimension, then records of [24-byte key][dimension x float32].
Records are appended in batches, so an interrupted indexing run resumes where it stopped.
The key is a hash of the passage text, so changed text is simply re-embedded.
The vectors are numbers derived from patient text: keep this folder on the encrypted
disk, like the PDFs and the database.
"""

from __future__ import annotations

import re
import struct
from array import array
from pathlib import Path

MAGIC = b"IMEV1"
KEY_LEN = 24


def index_path(index_dir: str | Path, case_id: str, model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return Path(index_dir) / case_id / f"{safe_model}.vec"


class DiskVectorCache(dict):
    """A dict of passage-hash -> vector that loads from and appends to one file."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.dim: int | None = None
        self._pending: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if not data.startswith(MAGIC) or len(data) < len(MAGIC) + 4:
            self.path.unlink()          # unreadable file: rebuild it
            return
        self.dim = struct.unpack_from("<I", data, len(MAGIC))[0]
        rec = KEY_LEN + 4 * self.dim
        pos = len(MAGIC) + 4
        while pos + rec <= len(data):    # a half-written last record (crash) is ignored
            key = data[pos:pos + KEY_LEN].decode("ascii")
            vec = array("f")
            vec.frombytes(data[pos + KEY_LEN:pos + rec])
            dict.__setitem__(self, key, vec.tolist())
            pos += rec

    def __setitem__(self, key: str, vec: list[float]) -> None:
        dict.__setitem__(self, key, vec)
        self._pending[key] = vec

    def flush(self) -> None:
        if not self._pending:
            return
        dim = len(next(iter(self._pending.values())))
        if self.dim is not None and dim != self.dim:      # model changed size: start the file over
            self.path.unlink(missing_ok=True)
            self._pending = {k: v for k, v in self.items() if len(v) == dim}
            self.dim = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        with self.path.open("ab") as f:
            if new_file:
                f.write(MAGIC + struct.pack("<I", dim))
                self.dim = dim
            for key, vec in self._pending.items():
                f.write(key.encode("ascii")[:KEY_LEN].ljust(KEY_LEN, b"_"))
                f.write(array("f", vec).tobytes())
        self._pending = {}
