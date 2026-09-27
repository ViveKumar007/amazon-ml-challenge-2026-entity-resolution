"""Candidate generation (blocking): for each Source 1 record, pick the few Source 2/3
records worth comparing.

Every record gets up to ~9 blocking keys of 7 types. Two records become a candidate
pair when they share a key:

    name2          the 2 rarest words of the cleaned name (name_core + name_alt)
    phon1          the rarest sound-key word (name_phon), for rough transliterations
    nospace8       first 8 letters of the name without spaces (domain-style names)
    addr_num_word  an address number (first 2) + the rarest address word
    addr_2words    the 2 rarest address words together (street + locality)
    name_full      the whole cleaned name as a word set (any word order)
    name_addr      rarest name word + rarest address word
    num_addrword   first address number + each of the 4 rarest address words
    nameword_addrword  rarest name word + 2nd / 3rd rarest address word
    nameword_num   rarest name word + first address number

The last three (v3) exist because typos usually hit the street name, which is the
rarest address word: "3060 manley ave springfield" vs "3060 mnaley ave springfield"
still share 3060 + springfield.

"Rarest" uses word frequencies counted per country over all three sources of the
split. Keys include the country, so records only pair within their own country.
A key is skipped when it is shared by more than BLOCK_CAP_S1 Source 1 records or
BLOCK_CAP_OTHER Source 2/3 records, or forms a block of more than BLOCK_MAX_PAIRS
pairs: generic words like "care" would otherwise create billions of pairs.

Every surviving pair is scored by a small LightGBM ranker (key matches + fast string
similarities, trained on spare train S1 records outside the train/val splits) and the
best TOP_K per Source 1 record are kept. That list is candidate_pairs.tsv and the
input of the matching model.

Keys are cached per split (<cache>/{split}_keys_v*.npz); everything is integer hashes
and numpy so it fits in 16 GB of RAM.

Output (<cache>/{val,train,test}_candidates.parquet):
    id1, id2, the RANK_FEATURES columns, key_mask (bit per key type),
    block_score (ranker probability), rank (0 = best candidate of that id1)

Run:  python src/blocking.py                    val + train + test
      python src/blocking.py --splits val       quick recall check only
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

import config
from io_utils import load_gt_pairs
from normalize import norm_path

KEY_TYPES = ["name2", "phon1", "nospace8", "addr_num_word", "addr_2words", "name_full", "name_addr",
             "num_addrword", "nameword_addrword", "nameword_num"]
KEYS_VERSION = 3          # bump when key definitions change, so cached keys are rebuilt
TEXT_COLS = ["name_core", "name_alt", "name_nospace", "addr_clean"]
LOAD_COLS = ["id", "country", "name_core", "name_alt", "name_nospace", "name_phon", "addr_clean"]
EVAL_KS = (5, 10, 20, 30, 50)
_HASH_KEY = "blocking-keys-01"            # pandas hash_array needs a 16-character key
_MASK64 = (1 << 64) - 1
_SALT = [np.uint64(((i + 1) * 0x9E3779B97F4A7C15) & _MASK64) for i in range(len(KEY_TYPES))]


# --------------------------------------------------------------------------- hashing helpers

def hstr(values):
    """64-bit hash of each string (numpy object array or pyarrow array)."""
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        values = values.to_numpy(zero_copy_only=False)
    return pd.util.hash_array(np.asarray(values, dtype=object), hash_key=_HASH_KEY, categorize=True)


def mix(a, b):
    """Combine two uint64 hashes (arrays or scalars) into one; order matters."""
    with np.errstate(over="ignore"):
        x = np.asarray(a, dtype=np.uint64) ^ (np.asarray(b, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))
        x ^= x >> np.uint64(29)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(32)
    return x


def _col(table, name):
    """One column of a pyarrow table as a single contiguous Array."""
    return table.column(name).combine_chunks()


def _split(arr):
    """Whitespace-split a pyarrow string array -> (row index per token, token array)."""
    lists = pc.utf8_split_whitespace(arr)
    tokens = pc.list_flatten(lists)
    rows = pc.list_parent_indices(lists).to_numpy()
    keep = pc.greater(pc.utf8_length(tokens), 0)
    return rows[keep.to_numpy(zero_copy_only=False)], tokens.filter(keep)


def _group_rank(g):
    """Position of each element inside its run of equal values (g must be sorted)."""
    n = len(g)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.arange(n)
    start = np.empty(n, dtype=bool)
    start[0] = True
    start[1:] = g[1:] != g[:-1]
    return idx - np.maximum.accumulate(np.where(start, idx, 0))


def _dedupe(rows, keys, *extra):
    """Sort by (row, key) and drop repeated (row, key) pairs; extra arrays follow along."""
    order = np.lexsort((keys, rows))
    rows, keys = rows[order], keys[order]
    keep = np.ones(len(rows), dtype=bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (keys[1:] != keys[:-1])
    return (rows[keep], keys[keep], *(e[order][keep] for e in extra))


# --------------------------------------------------------------------------- per-record keys

def _fields(table):
    """Hashed tokens of each field for one source table: {field: (rows, keys)}.

    Word keys are mixed with the country hash so identical words in different
    countries never share a key (and never share a frequency count).
    """
    country = hstr(_col(table, "country"))
    out = {}
    names = pc.binary_join_element_wise(_col(table, "name_core"), _col(table, "name_alt"), " ")
    for field, arr in (("name", names), ("phon", _col(table, "name_phon")), ("core", _col(table, "name_core"))):
        r, tok = _split(arr)
        out[field] = _dedupe(r, mix(hstr(tok), country[r]))

    r, tok = _split(_col(table, "addr_clean"))
    alpha = pc.and_(pc.utf8_is_alpha(tok), pc.greater(pc.utf8_length(tok), 2))
    a = alpha.to_numpy(zero_copy_only=False)
    out["addr"] = _dedupe(r[a], mix(hstr(tok.filter(alpha)), country[r[a]]))
    digit = pc.utf8_is_digit(tok)
    d = digit.to_numpy(zero_copy_only=False)
    rd, hd = r[d], hstr(tok.filter(digit))            # rows ascending, tokens in address order
    first2 = _group_rank(rd) < 2
    out["num"] = (rd[first2], hd[first2])
    return out, country


def _add_counts(acc, keys):
    """Add document-frequency counts of keys (unique per record) to acc=(sorted keys, counts)."""
    u, c = np.unique(keys, return_counts=True)
    if acc is None:
        return u, c
    allu, inv = np.unique(np.concatenate([acc[0], u]), return_inverse=True)
    return allu, np.bincount(inv, weights=np.concatenate([acc[1], c])).astype(np.int64)


def _lookup(acc, keys):
    """Document frequency of each key (every key must be present in acc)."""
    return acc[1][np.searchsorted(acc[0], keys)]


def _rarest(rows, keys, df, k):
    """Keep the k lowest-frequency keys per row -> (rows, keys, rank), sorted by row."""
    order = np.lexsort((keys, df, rows))
    rows, keys = rows[order], keys[order]
    rank = _group_rank(rows)
    keep = rank < k
    return rows[keep], keys[keep], rank[keep]


def _join_rows(ra, ka, rb, kb):
    """Pair every (row, key) of a with the key of the same row in b (rb sorted, unique).

    Returns (rows, (keys_a, keys_b)) for the rows present in both.
    """
    if len(rb) == 0 or len(ra) == 0:
        return ra[:0], (ka[:0], kb[:0])
    pos = np.minimum(np.searchsorted(rb, ra), len(rb) - 1)
    ok = rb[pos] == ra
    return ra[ok], (ka[ok], kb[pos[ok]])


def record_keys(table, dfreq):
    """All blocking keys of one source table -> (rows, keys, key_type)."""
    fields, country = _fields(table)
    out_r, out_k, out_t = [], [], []

    def add(r, k, t):
        out_r.append(r)
        out_k.append(mix(k, _SALT[t]))
        out_t.append(np.full(len(r), t, dtype=np.int8))

    r, k = fields["name"]
    r, k, rank = _rarest(r, k, _lookup(dfreq["name"], k), 2)
    add(r, k, 0)
    name0_r, name0_k = r[rank == 0], k[rank == 0]            # rarest name word, one per row

    # full name as a word SET (order-free): sum of the word hashes of each row
    r, k = fields["core"]
    if len(r):
        starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        with np.errstate(over="ignore"):
            add(r[starts], np.add.reduceat(k, starts), 5)

    r, k = fields["phon"]
    r, k, _ = _rarest(r, k, _lookup(dfreq["phon"], k), 1)
    add(r, k, 1)

    nospace = _col(table, "name_nospace")
    ok = pc.greater_equal(pc.utf8_length(nospace), 6)
    r = np.flatnonzero(ok.to_numpy(zero_copy_only=False))
    add(r, mix(hstr(pc.utf8_slice_codeunits(nospace.filter(ok), 0, 8)), country[r]), 2)

    r, k = fields["addr"]
    r, k, rank = _rarest(r, k, _lookup(dfreq["addr"], k), 4)
    wr, wk = r[rank == 0], k[rank == 0]                     # rarest address word, one per row
    num_r, num_h = fields["num"]                            # first 2 numbers per row
    rr, kk = _join_rows(num_r, num_h, wr, wk)
    add(rr, mix(kk[1], kk[0]), 3)

    second = np.flatnonzero(rank == 1)                      # previous element = rank 0 of same row
    a, b = k[second - 1], k[second]
    add(r[second], mix(np.minimum(a, b), np.maximum(a, b)), 4)

    # rarest name word + rarest address word (a common name made specific by its place)
    both, i, j = np.intersect1d(name0_r, wr, assume_unique=True, return_indices=True)
    add(both, mix(name0_k[i], wk[j]), 6)

    # v3: typo-tolerant address keys built from the first number and several words
    first = _group_rank(num_r) == 0
    fn_r, fn_h = num_r[first], num_h[first]                 # first address number, one per row
    rr, kk = _join_rows(r, k, fn_r, fn_h)                   # each of the 4 rarest words x number
    add(rr, mix(kk[0], kk[1]), 7)
    sel = (rank == 1) | (rank == 2)
    rr, kk = _join_rows(r[sel], k[sel], name0_r, name0_k)   # 2nd/3rd rarest word x name word
    add(rr, mix(kk[1], kk[0]), 8)
    rr, kk = _join_rows(name0_r, name0_k, fn_r, fn_h)       # name word x number
    add(rr, mix(kk[0], kk[1]), 9)

    rows, keys, types = _dedupe(np.concatenate(out_r), np.concatenate(out_k), np.concatenate(out_t))
    return rows, keys, types


def _load(split, source):
    """Load the normalised columns blocking needs, as one contiguous pyarrow table."""
    return pq.read_table(norm_path(split, source), columns=LOAD_COLS).combine_chunks()


def keys_path(split, side, cache_dir=config.CACHE_DIR):
    """Cached key arrays of one split; side is 's1' or 'other'."""
    return Path(cache_dir) / f"{split}_keys_v{KEYS_VERSION}_{side}.npz"


def build_keys(split):
    """Compute blocking keys for all records of a split and cache them to disk.

    One source file is in memory at a time. S2 and S3 are stacked into one 'other'
    table (S3 rows follow S2 rows) and its keys are stored sorted for fast lookup.
    """
    t0 = time.time()
    dfreq = {"name": None, "phon": None, "addr": None}
    for s in (1, 2, 3):                       # pass 1: word frequencies per country
        fields, _ = _fields(_load(split, s))
        for f in dfreq:
            dfreq[f] = _add_counts(dfreq[f], fields[f][1])
        del fields
        gc.collect()
    print(f"  [{split}] word frequencies: {time.time() - t0:.0f}s", flush=True)

    keys, offset = {}, 0
    for s in (1, 2, 3):                       # pass 2: keys per record
        t = _load(split, s)
        r, k, ty = record_keys(t, dfreq)
        keys[s] = (r + (offset if s == 3 else 0), k, ty)
        offset = t.num_rows if s == 2 else offset
        del t
        gc.collect()
    np.savez(keys_path(split, "s1"), rows=keys[1][0], keys=keys[1][1], types=keys[1][2])
    k = np.concatenate([keys[2][1], keys[3][1]])
    order = np.argsort(k, kind="stable")
    rows = np.concatenate([keys[2][0], keys[3][0]])[order]
    np.savez(keys_path(split, "other"), rows=rows, keys=k[order])
    print(f"  [{split}] keys built and cached: {time.time() - t0:.0f}s", flush=True)


class SplitIndex:
    """Blocking keys and texts of one split (all S1 + all S2/S3 records of train or test)."""

    def __init__(self, split, rebuild=False):
        t0 = time.time()
        if rebuild or not (keys_path(split, "s1").exists() and keys_path(split, "other").exists()):
            build_keys(split)

        s1 = np.load(keys_path(split, "s1"))
        self.s1_rows, self.s1_keys, self.s1_types = s1["rows"], s1["keys"], s1["types"]
        self.k1_u, self.k1_c = np.unique(self.s1_keys, return_counts=True)   # S1 records per key
        other = np.load(keys_path(split, "other"))
        self.o_rows, self.o_keys = other["rows"].astype(np.int32), other["keys"]

        cols = ["id", "country"] + TEXT_COLS
        t1 = pq.read_table(norm_path(split, 1), columns=cols).combine_chunks()
        to = pa.concat_tables([pq.read_table(norm_path(split, s), columns=cols) for s in (2, 3)]).combine_chunks()
        self.split = split
        self.s1_ids = _col(t1, "id").to_numpy()
        self.s1_country = _col(t1, "country").to_numpy(zero_copy_only=False)
        self.o_ids = _col(to, "id").to_numpy()
        self.s1_text = {c: _col(t1, c) for c in TEXT_COLS}
        self.o_text = {c: _col(to, c) for c in TEXT_COLS}
        del t1, to
        gc.collect()
        print(f"  [{split}] index ready: {len(self.s1_keys):,} S1 keys, {len(self.o_keys):,} S2/S3 keys "
              f"({time.time() - t0:.0f}s)", flush=True)

    def rows_of(self, ids):
        """Row positions of Source 1 ids in this index."""
        pos = pd.Index(self.s1_ids).get_indexer(ids)
        if (pos < 0).any():
            raise ValueError("some ids are not Source 1 records of this split")
        return pos


# --------------------------------------------------------------------------- candidate pairs

def _take(arr, idx):
    """Strings at positions idx of a pyarrow string array, as a numpy object array.

    (to_numpy is ~50x faster than to_pylist here.)
    """
    return arr.take(pa.array(idx)).to_numpy(zero_copy_only=False)


RANK_FEATURES = [f"key_{t}" for t in KEY_TYPES] + [
    "n_keys", "n_key_types", "name_tset", "name_tsort", "name_nospace", "alt_tset",
    "addr_tset", "addr_tsort", "o_addr_empty", "name_best_gap",
]


def pair_features(ix, q, o, n_keys, mask):
    """Cheap features of candidate pairs (S1 row q, S2/S3 row o), used to rank them.

    All similarities are rapidfuzz scores 0-100 on the normalised text. name_best_gap
    compares a pair with the best-named candidate of the same S1 record, so every
    candidate of an S1 record must be in the same call (true for candidate_chunks).
    """
    sim = lambda a, b, scorer: process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)
    X = pd.DataFrame({f"key_{t}": ((mask >> i) & 1).astype(np.int8) for i, t in enumerate(KEY_TYPES)})
    X["n_keys"] = n_keys
    X["n_key_types"] = X[[f"key_{t}" for t in KEY_TYPES]].sum(axis=1).astype(np.int8)
    q_core, o_core = _take(ix.s1_text["name_core"], q), _take(ix.o_text["name_core"], o)
    X["name_tset"] = sim(q_core, o_core, fuzz.token_set_ratio)
    X["name_tsort"] = sim(q_core, o_core, fuzz.token_sort_ratio)
    X["name_nospace"] = sim(_take(ix.s1_text["name_nospace"], q), _take(ix.o_text["name_nospace"], o), fuzz.ratio)
    X["alt_tset"] = sim(q_core, _take(ix.o_text["name_alt"], o), fuzz.token_set_ratio)
    del q_core, o_core
    o_addr = ix.o_text["addr_clean"].take(pa.array(o))
    q_addr, o_addr_np = _take(ix.s1_text["addr_clean"], q), o_addr.to_numpy(zero_copy_only=False)
    X["addr_tset"] = sim(q_addr, o_addr_np, fuzz.token_set_ratio)
    X["addr_tsort"] = sim(q_addr, o_addr_np, fuzz.token_sort_ratio)
    X["o_addr_empty"] = (pc.utf8_length(o_addr).to_numpy() == 0).astype(np.int8)
    X["name_best_gap"] = X.groupby(q)["name_tset"].transform("max").to_numpy() - X["name_tset"]
    return X


def ranker_path(cache_dir=config.CACHE_DIR):
    """Saved LightGBM model that ranks candidates (tied to the key definitions)."""
    return Path(cache_dir) / f"blocking_ranker_v{KEYS_VERSION}.txt"


def train_ranker(ix, n_s1=20_000):
    """Train the candidate ranker on train S1 records outside the train/val splits.

    Those records are never used for matching-model training or validation, so the
    ranker cannot leak labels into later scores.
    """
    import lightgbm as lgb
    from split import SPLIT_FILE

    t0 = time.time()
    used = pd.read_parquet(Path(config.CACHE_DIR) / SPLIT_FILE, columns=["id1"])["id1"]
    spare = np.setdiff1d(ix.s1_ids, used.to_numpy())
    ids = np.random.default_rng(config.SEED).choice(spare, n_s1, replace=False)
    rows = np.sort(ix.rows_of(ids))
    q, o, n_keys, mask = raw_pairs(ix, rows, _query_keys(ix, rows))
    X = pair_features(ix, q, o, n_keys, mask)

    gt = load_gt_pairs()
    gt = gt[gt["id1"].isin(ids)]
    n_other = len(ix.o_ids)
    gt_code = ix.rows_of(gt["id1"].to_numpy()) * n_other + pd.Index(ix.o_ids).get_indexer(gt["id2"].to_numpy())
    y = np.isin(q * n_other + o, gt_code)

    model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                               random_state=config.SEED, verbose=-1)
    model.fit(X[RANK_FEATURES], y)
    model.booster_.save_model(str(ranker_path()))
    print(f"  ranker trained on {len(y):,} pairs from {n_s1:,} spare S1 "
          f"({y.mean():.2%} true) in {time.time() - t0:.0f}s -> {ranker_path().name}", flush=True)


def load_ranker():
    """Load the saved candidate ranker."""
    import lightgbm as lgb
    return lgb.Booster(model_file=str(ranker_path()))


def _query_keys(ix, query_rows):
    """Keys of the given Source 1 rows, sorted by row -> (rows, keys, types)."""
    sel = np.isin(ix.s1_rows, query_rows)
    return ix.s1_rows[sel], ix.s1_keys[sel], ix.s1_types[sel]


def raw_pairs(ix, rows, qkeys, cap_s1=config.BLOCK_CAP_S1, cap_other=config.BLOCK_CAP_OTHER,
              max_block=config.BLOCK_MAX_PAIRS):
    """All distinct candidate pairs of a chunk of Source 1 rows -> (q, o, n_keys, key_mask).

    A key is skipped when it is shared by more than cap_s1 Source 1 records, more than
    cap_other Source 2/3 records, or when the block it forms has more than max_block pairs.
    """
    qr, qk, qt = qkeys
    lo, hi = np.searchsorted(qr, rows[0], "left"), np.searchsorted(qr, rows[-1], "right")
    r, k, t = qr[lo:hi], qk[lo:hi], qt[lo:hi]

    n1 = ix.k1_c[np.searchsorted(ix.k1_u, k)]
    left, right = np.searchsorted(ix.o_keys, k, "left"), np.searchsorted(ix.o_keys, k, "right")
    n2 = right - left
    ok = (n2 > 0) & (n1 <= cap_s1) & (n2 <= cap_other) & (n1 * n2 <= max_block)
    r, t, left, n2 = r[ok], t[ok], left[ok], n2[ok]

    # expand every (S1 key, matching S2/S3 records) block into pairs
    total = int(n2.sum())
    q = np.repeat(r, n2)
    pos = np.arange(total) - np.repeat(np.cumsum(n2) - n2, n2) + np.repeat(left, n2)
    o = ix.o_rows[pos]
    bits = np.repeat((1 << t.astype(np.int16)), n2)

    # one row per distinct (S1, S2/S3) pair: count shared keys, OR the key-type bits
    n_other = len(ix.o_ids)
    code = q * n_other + o
    order = np.argsort(code, kind="stable")
    code, bits = code[order], bits[order]
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty.astype(np.int16), empty.astype(np.int16)
    starts = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
    n_keys = np.diff(np.r_[starts, total]).astype(np.int16)
    mask = np.bitwise_or.reduceat(bits, starts).astype(np.int16)
    return code[starts] // n_other, code[starts] % n_other, n_keys, mask


def candidate_chunks(ix, query_rows, keep_k, ranker, chunk=config.BLOCK_CHUNK, min_score=config.BLOCK_MIN_SCORE,
                     **caps):
    """Yield (candidates DataFrame, n raw pairs per query row) for chunks of Source 1 rows.

    Every raw pair is scored by the ranker; the best keep_k per Source 1 row are kept,
    with their features, the ranker score (block_score) and rank (0 = best).
    """
    query_rows = np.sort(np.asarray(query_rows))
    qkeys = _query_keys(ix, query_rows)
    for start in range(0, len(query_rows), chunk):
        rows = query_rows[start:start + chunk]
        q, o, n_keys, mask = raw_pairs(ix, rows, qkeys, **caps)
        X = pair_features(ix, q, o, n_keys, mask)
        score = ranker.predict(X[RANK_FEATURES]).astype(np.float32)
        order = np.lexsort((-score, q))
        rank = _group_rank(q[order])
        # at most keep_k per S1, and only pairs the ranker gives a real chance (BLOCK_MIN_SCORE)
        top = (rank < keep_k) & (score[order] >= min_score)
        keep, rank = order[top], rank[top]
        cand = X.iloc[keep].reset_index(drop=True)
        cand.insert(0, "o", o[keep])
        cand.insert(0, "q", q[keep])
        cand["key_mask"] = mask[keep]
        cand["block_score"] = score[keep]
        cand["rank"] = rank.astype(np.int16)
        raw_per_row = np.bincount(np.searchsorted(rows, q), minlength=len(rows))
        del X, score
        yield cand, raw_per_row


def _to_output(ix, cand):
    """Replace row positions with entity ids (the saved candidate format)."""
    out = cand.drop(columns=["q", "o"])
    out.insert(0, "id2", ix.o_ids[cand["o"].to_numpy()])
    out.insert(0, "id1", ix.s1_ids[cand["q"].to_numpy()])
    return out


def cand_path(name, cache_dir=config.CACHE_DIR):
    """Path of a saved candidate file (name = val, train or test)."""
    return Path(cache_dir) / f"{name}_candidates.parquet"


def load_candidates(name, cache_dir=config.CACHE_DIR):
    """Load saved candidates, e.g. load_candidates("val")."""
    return pd.read_parquet(cand_path(name, cache_dir))


# --------------------------------------------------------------------------- runs

def run_labelled(ix, name, query_ids, ranker, k=config.TOP_K):
    """Block a labelled set of train S1 ids, report recall, save the top-k candidates.

    val keeps up to max(EVAL_KS) candidates in memory to report recall at several K;
    train keeps only k (it is 4x larger).
    """
    t0 = time.time()
    rows = ix.rows_of(query_ids)
    kmax = max(max(EVAL_KS), k) if name == "val" else k
    parts, raw = [], []
    for cand, raw_per_row in candidate_chunks(ix, rows, kmax, ranker):
        parts.append(cand)
        raw.append(raw_per_row)
    cand = pd.concat(parts, ignore_index=True)
    raw = np.concatenate(raw)

    gt = load_gt_pairs()
    gt = gt[gt["id1"].isin(query_ids)]
    n_other = len(ix.o_ids)
    gt_code = np.sort(ix.rows_of(gt["id1"].to_numpy()) * n_other
                      + pd.Index(ix.o_ids).get_indexer(gt["id2"].to_numpy()))
    code = cand["q"].to_numpy() * n_other + cand["o"].to_numpy()
    cand["is_true"] = np.isin(code, gt_code)

    n_true = len(gt_code)
    country = pd.Series(ix.s1_country[cand["q"].to_numpy()])
    gt_country = pd.Series(ix.s1_country[ix.rows_of(gt["id1"].to_numpy())]).value_counts()
    print(f"\n[{name}] {len(rows):,} S1 records, {n_true:,} true pairs  ({time.time() - t0:.0f}s)")
    print(f"  raw candidate pairs per S1: mean {raw.mean():.1f}, median {np.median(raw):.0f}, "
          f"p95 {np.percentile(raw, 95):.0f}; S1 with no candidate: {(raw == 0).mean():.2%}")
    for kk in [x for x in EVAL_KS if x <= kmax]:
        top = cand["rank"] < kk
        by_c = (cand[top & cand["is_true"]].groupby(country[top & cand["is_true"]]).size() / gt_country)
        print(f"  recall@{kk:<3} {cand.loc[top, 'is_true'].sum() / n_true:.4f}   "
              + "  ".join(f"{c} {v:.4f}" for c, v in by_c.items())
              + f"   ({top.sum() / len(rows):.1f} pairs/S1)")
    hits = cand.loc[cand["is_true"] & (cand["rank"] < k), "key_mask"].to_numpy()
    print("  share of found true pairs sharing each key type: "
          + ", ".join(f"{t} {((hits >> i) & 1).mean():.3f}" for i, t in enumerate(KEY_TYPES)))

    _write_misses(ix, name, gt, cand[cand["rank"] < k])
    out = _to_output(ix, cand[cand["rank"] < k].drop(columns="is_true"))
    out.to_parquet(cand_path(name), index=False)
    print(f"  saved {len(out):,} candidates -> {cand_path(name).name}")
    return out


def _write_misses(ix, name, gt, cand, n=200):
    """Write a sample of true pairs that did not make the top-k list, for manual review."""
    found = set(zip(ix.s1_ids[cand.loc[cand["is_true"], "q"]], ix.o_ids[cand.loc[cand["is_true"], "o"]]))
    miss = gt[[(a, b) not in found for a, b in zip(gt["id1"], gt["id2"])]]
    miss = miss.sample(min(n, len(miss)), random_state=config.SEED)
    ids = np.unique(np.r_[miss["id1"].to_numpy(), miss["id2"].to_numpy()]).tolist()
    raw = pd.concat([pq.read_table(Path(config.CACHE_DIR) / f"train_source{s}.parquet",
                                   columns=["id", "business_name", "business_address"],
                                   filters=[("id", "in", ids)]).to_pandas()
                     for s in (1, 2, 3)]).set_index("id")
    rows = [(raw.loc[a, "business_name"], raw.loc[b, "business_name"],
             raw.loc[a, "business_address"], raw.loc[b, "business_address"]) for a, b in zip(miss["id1"], miss["id2"])]
    path = Path(config.CACHE_DIR) / f"{name}_blocking_misses.tsv"
    pd.DataFrame(rows, columns=["s1_name", "other_name", "s1_address", "other_address"]).to_csv(
        path, sep="\t", index=False, encoding="utf-8")
    print(f"  {len(miss)} missed true pairs written to {path.name}")


SWEEP_GRID = [   # (cap_s1, cap_other, max_block)
    (50, 1000, 10**12), (200, 1000, 20_000), (500, 1000, 20_000), (500, 1000, 50_000),
    (1000, 2000, 100_000), (2000, 3000, 300_000),
]


def sweep(ix, query_ids, grid=SWEEP_GRID, chunk=5_000):
    """Recall ceiling and pair count of the raw candidates (before top-K) for several limits."""
    rows = np.sort(ix.rows_of(query_ids))
    qkeys = _query_keys(ix, rows)
    gt = load_gt_pairs()
    gt = gt[gt["id1"].isin(query_ids)]
    n_other = len(ix.o_ids)
    gt_code = np.sort(ix.rows_of(gt["id1"].to_numpy()) * n_other
                      + pd.Index(ix.o_ids).get_indexer(gt["id2"].to_numpy()))
    print(f"\nSweep on {len(rows):,} S1 / {len(gt_code):,} true pairs (raw pairs, before top-K)")
    print("  cap_s1 cap_other max_block   recall  pairs/S1   recall reached by each key type")
    for cap_s1, cap_other, max_block in grid:
        t0, found, n_pairs, hits = time.time(), 0, 0, np.zeros(len(KEY_TYPES))
        for start in range(0, len(rows), chunk):
            q, o, _, mask = raw_pairs(ix, rows[start:start + chunk], qkeys,
                                      cap_s1=cap_s1, cap_other=cap_other, max_block=max_block)
            is_true = np.isin(q * n_other + o, gt_code)
            found += int(is_true.sum())
            n_pairs += len(q)
            hits += [((mask[is_true] >> i) & 1).sum() for i in range(len(KEY_TYPES))]
        print(f"  {cap_s1:>6} {cap_other:>9} {max_block:>9.0e}   {found / len(gt_code):.4f} "
              f"{n_pairs / len(rows):>9.1f}   "
              + " ".join(f"{t}={h / len(gt_code):.2f}" for t, h in zip(KEY_TYPES, hits))
              + f"  ({time.time() - t0:.0f}s)", flush=True)


def run_stream(ix, ranker, name, k=config.TOP_K):
    """Block EVERY S1 record of the index and stream the top-k candidates to parquet.

    Used for test, and for all train S1 ("train_all") so that competition between S1
    records claiming the same S2/S3 record looks the same in training as on test.
    The full top-k is written to <name>_candidates_top20.parquet (input of union.py), then
    shrink() applies BLOCK_MIN_SCORE to give <name>_candidates.parquet.
    """
    t0 = time.time()
    full = cand_path(name).with_name(f"{name}_candidates_top20.parquet")
    rows = np.arange(len(ix.s1_ids))
    writer, raw, n_out = None, [], 0
    try:
        for cand, raw_per_row in candidate_chunks(ix, rows, k, ranker, min_score=0.0):
            table = pa.Table.from_pandas(_to_output(ix, cand), preserve_index=False)
            writer = writer or pq.ParquetWriter(full, table.schema)
            writer.write_table(table)
            raw.append(raw_per_row)
            n_out += len(cand)
            done = sum(len(x) for x in raw)
            print(f"  [{name}] {done:,}/{len(rows):,} S1 ({time.time() - t0:.0f}s)", flush=True)
    finally:
        if writer:
            writer.close()
    raw = np.concatenate(raw)
    per = pd.Series(raw).groupby(ix.s1_country).agg(["mean", lambda x: (x == 0).mean()])
    print(f"\n[{name}] {len(rows):,} S1, {n_out:,} candidates saved ({n_out / len(rows):.1f}/S1) "
          f"in {time.time() - t0:.0f}s")
    for c, (m, z) in per.iterrows():
        print(f"  {c:<8} raw pairs/S1 {m:6.1f}   no candidate {z:.2%}")
    shrink(name)


def shrink(name, min_score=config.BLOCK_MIN_SCORE):
    """Apply BLOCK_MIN_SCORE to an already written candidate file (same result as re-blocking,
    since the ranker score of every kept pair is stored). The full top-K file is kept as
    <name>_candidates_top20.parquet."""
    src, backup = cand_path(name), cand_path(name).with_name(f"{name}_candidates_top20.parquet")
    if not backup.exists():
        src.rename(backup)
    c = pd.read_parquet(backup)
    out = c[c["block_score"] >= min_score]
    out.to_parquet(src, index=False)
    print(f"  [{name}] {len(c):,} -> {len(out):,} candidates ({len(out) / c['id1'].nunique():.2f} per S1 "
          f"that had candidates), min ranker score {min_score}", flush=True)


def recall_from_file(name, query_ids, label):
    """Recall@K of a saved candidate file, restricted to some labelled S1 ids."""
    cand = pd.read_parquet(cand_path(name), columns=["id1", "id2", "rank"],
                           filters=[("id1", "in", query_ids.tolist())])
    gt = load_gt_pairs()
    gt = gt[gt["id1"].isin(query_ids)]
    hit = cand.merge(gt, on=["id1", "id2"], how="inner")
    print(f"  [{label}] {len(query_ids):,} S1, {len(gt):,} true pairs: "
          + "  ".join(f"recall@{kk} {(hit['rank'] < kk).sum() / len(gt):.4f}" for kk in (5, 10, 20)))


def main(argv=None):
    """Command-line entry point."""
    from split import load_split

    ap = argparse.ArgumentParser(description="Blocking / candidate generation")
    ap.add_argument("--splits", nargs="+", default=["val", "train", "test"], choices=["val", "train", "test"])
    ap.add_argument("--sweep", action="store_true", help="only compare size limits on val (raw recall)")
    ap.add_argument("--rebuild", action="store_true", help="recompute cached keys")
    ap.add_argument("--retrain-ranker", action="store_true", help="retrain the candidate ranker")
    ap.add_argument("--all-train", action="store_true",
                    help="block ALL train S1 into train_all_candidates.parquet (instead of val/train)")
    ap.add_argument("--keys-only", action="store_true", help="only build and cache keys for --splits")
    ap.add_argument("--shrink", nargs="+", default=None,
                    help="apply BLOCK_MIN_SCORE to existing candidate files, e.g. --shrink train_all test")
    args = ap.parse_args(argv)
    if args.shrink:
        for name in args.shrink:
            shrink(name)
        return
    print(f"keys v{KEYS_VERSION}; cap S1 {config.BLOCK_CAP_S1}, cap S2/S3 {config.BLOCK_CAP_OTHER}, "
          f"max block {config.BLOCK_MAX_PAIRS}, top-K {config.TOP_K}", flush=True)

    if args.keys_only:
        for split in sorted({"test" if s == "test" else "train" for s in args.splits}):
            build_keys(split)
        return
    if args.sweep:
        sweep(SplitIndex("train", rebuild=args.rebuild), load_split("val"))
        return
    labelled = [] if args.all_train else [s for s in args.splits if s in ("val", "train")]
    need_ranker = args.retrain_ranker or not ranker_path().exists()
    if labelled or need_ranker or args.all_train:
        ix = SplitIndex("train", rebuild=args.rebuild)
        if need_ranker:
            train_ranker(ix)
        ranker = load_ranker()
        for name in labelled:
            run_labelled(ix, name, load_split(name), ranker)
        if args.all_train:
            run_stream(ix, ranker, "train_all")
            for name in ("val", "train"):
                recall_from_file("train_all", load_split(name), name)
        del ix
        gc.collect()
    if "test" in args.splits:
        run_stream(SplitIndex("test", rebuild=args.rebuild), load_ranker(), "test")


if __name__ == "__main__":
    sys.exit(main())
