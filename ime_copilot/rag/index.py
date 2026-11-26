"""Search index over a case file's pages.

- Pages are split into overlapping passages (~200 words); every passage keeps
  its page number, so every search hit maps back to a page.
- Keyword search: BM25, with tokens that keep medical codes intact
  (97110, M54.16, 64483-LT) and a small expansion of common abbreviations
  (PT -> physical therapy, ESI -> epidural steroid injection ...).
- Optional meaning-based search: embeddings from Azure OpenAI, combined with
  BM25 by reciprocal rank fusion (hybrid search).
- Duplicate pages (faxed copies) are indexed once.

Keyword index: built in memory, a few seconds for 2,000 pages.
Meaning vectors: computed once and saved on disk (rag/store.py), then reused.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

try:                                   # optional: much faster vector search when installed
    import numpy as _np
except ImportError:                    # pragma: no cover
    _np = None

from ..schemas import Page

PASSAGE_WORDS = 200
PASSAGE_STRIDE = 150

_TOKEN = re.compile(r"[a-z]\d{2}\.[0-9a-z]{1,4}|\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_STOP = set("""a an the and or of to in on at for by with from as is are was were be been being this that these
those it its into than then there their they he she his her him we our you your i me my not no yes do does did
has have had will would can could should may might shall what which who whom whose when where why how all any
each per please""".split())

# Abbreviations used in claim files and in questions, expanded into the words notes use.
EXPANSIONS: dict[str, str] = {
    "pt": "physical therapy",
    "ot": "occupational therapy",
    "esi": "epidural steroid injection",
    "tfesi": "transforaminal epidural steroid injection",
    "mri": "mri magnetic resonance imaging",
    "ct": "ct computed tomography",
    "emg": "emg electromyography",
    "ncs": "nerve conduction study",
    "rfa": "radiofrequency ablation",
    "mbb": "medial branch block",
    "ov": "office visit",
    "mmi": "maximum medical improvement",
    "rtw": "return to work",
    "rom": "range of motion",
    "odi": "oswestry disability index",
    "dos": "date of service",
    "doi": "date of injury",
    "mva": "motor vehicle accident",
    "mvc": "motor vehicle collision",
    "hx": "history",
    "dx": "diagnosis",
    "tx": "treatment",
    "rx": "prescription medication",
    "inj": "injection",
    "chiro": "chiropractic",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text) if t.lower() not in _STOP]


def expand_query(question: str) -> list[str]:
    tokens = tokenize(question)
    extra = [w for t in tokens if t in EXPANSIONS for w in tokenize(EXPANSIONS[t])]
    return tokens + extra


@dataclass
class Passage:
    pid: int
    page_no: int
    text: str


@dataclass
class Hit:
    passage: Passage
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None


def make_passages(pages: Sequence[Page]) -> list[Passage]:
    out: list[Passage] = []
    for page in pages:
        if page.duplicate_of is not None or not page.text.strip():
            continue
        words = page.text.split()
        if len(words) <= PASSAGE_WORDS:
            out.append(Passage(len(out), page.page_no, page.text.strip()))
            continue
        # Split on line boundaries where possible so quotes stay intact.
        start = 0
        while start < len(words):
            chunk = " ".join(words[start:start + PASSAGE_WORDS])
            out.append(Passage(len(out), page.page_no, chunk))
            if start + PASSAGE_WORDS >= len(words):
                break
            start += PASSAGE_STRIDE
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = Counter(query)
        out = []
        for d, length in zip(self.docs, self.lens):
            s = 0.0
            for t, qf in q.items():
                tf = d.get(t)
                if not tf:
                    continue
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
                s += self.idf.get(t, 0.0) * norm * (1 + 0.3 * (qf - 1))
            out.append(s)
        return out


Embedder = Callable[[list[str]], list[list[float]]]


@dataclass
class CaseIndex:
    passages: list[Passage]
    bm25: BM25
    vectors: list[list[float]] | None = None
    embedder: Embedder | None = None
    pages: dict[int, Page] = field(default_factory=dict)

    @classmethod
    def build(cls, pages: Sequence[Page], embedder: Embedder | None = None,
              vector_cache: dict[str, list[float]] | None = None,
              on_progress: Callable[[int, int], None] | None = None) -> "CaseIndex":
        passages = make_passages(pages)
        bm25 = BM25([tokenize(p.text) for p in passages])
        vectors = None
        if embedder is not None and passages:
            vectors = _embed_with_cache(embedder, [p.text for p in passages], vector_cache,
                                        on_progress=on_progress)
            vectors = [_unit(v) for v in vectors]
        idx = cls(passages, bm25, vectors, embedder, {p.page_no: p for p in pages})
        idx._matrix = _np.asarray(vectors, dtype="float32") if (_np is not None and vectors) else None
        return idx

    def search(self, question: str, k: int = 8) -> list[Hit]:
        if not self.passages:
            return []
        bm = self.bm25.scores(expand_query(question))
        bm_order = sorted(range(len(bm)), key=lambda i: -bm[i])
        bm_rank = {i: r for r, i in enumerate(bm_order) if bm[i] > 0}

        vec_rank: dict[int, int] = {}
        if self.vectors is not None and self.embedder is not None:
            q = _unit(self.embedder([question])[0])
            matrix = getattr(self, "_matrix", None)
            if matrix is not None:
                sims = (matrix @ _np.asarray(q, dtype="float32")).tolist()
            else:
                sims = [sum(x * y for x, y in zip(q, v)) for v in self.vectors]
            vec_order = sorted(range(len(sims)), key=lambda i: -sims[i])
            vec_rank = {i: r for r, i in enumerate(vec_order[:200])}

        # Reciprocal rank fusion: robust way to merge keyword and meaning-based rankings.
        fused: dict[int, float] = {}
        for ranks in (bm_rank, vec_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.passages[i], round(fused[i], 6), bm_rank.get(i), vec_rank.get(i)) for i in top]


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _embed_with_cache(embedder: Embedder, texts: list[str], cache: dict[str, list[float]] | None,
                      batch: int = 64, on_progress: Callable[[int, int], None] | None = None
                      ) -> list[list[float]]:
    cache = cache if cache is not None else {}
    keys = [hashlib.sha256(t.encode()).hexdigest()[:24] for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    total, done = len(keys), len(keys) - len(missing)
    if on_progress:
        on_progress(done, total)
    for start in range(0, len(missing), batch):
        idx = missing[start:start + batch]
        for i, vec in zip(idx, embedder([texts[i] for i in idx])):
            cache[keys[i]] = vec
        if hasattr(cache, "flush"):
            cache.flush()                 # saved after every batch: an interrupted run resumes here
        done += len(idx)
        if on_progress:
            on_progress(done, total)
    return [cache[k] for k in keys]
