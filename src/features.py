"""Pair features for the matching model (step 5).

Input:  <cache>/{train_all,test}_candidates.parquet from union.py (or blocking.py), already
        holding the ranker features, block_score, rank and bienc_cos / bienc_rank.
Output: <cache>/{val,train,test}_features.parquet: the candidate columns plus the
        features below, and `label` (1 = true match) for val/train.

New features, all computed on the normalised text (normalize.py):

  name        name_ratio (whole-string edit similarity), name_partial_ns (one name
              inside the other, spaces removed: "earnosethroat" ~ "ear nose throat
              medicine"), name_jw_ns (Jaro-Winkler, spaces removed), phon_tset
              (sound-key words), name_idf_jacc / name_idf_max / name_idf_s1_missing
              (overlap weighted by word rarity: sharing "hypnosis" counts far more
              than sharing "care"), name_n1 / name_n2 (word counts), len1 / len2
  address     addr_idf_jacc / addr_idf_max (rarity-weighted overlap), num_jacc,
              num_first_eq (house number), num_any_eq, num_n1 / num_n2,
              postcode_eq and state_eq (1 same, 0 different, -1 unknown)
  record      is_s3, has_indic1 / has_indic2, alt_nonempty
  context     g_n_cand, g_n_strong (candidates of this S1 with block_score > 0.5),
              g_margin (block_score minus the best OTHER candidate of this S1),
              o_n_claims (how many S1 records have this S2/S3 record as candidate),
              o_margin (block_score minus the best OTHER S1 claiming this record).
              Each S2/S3 record belongs to at most one S1, so a negative o_margin
              means a better owner exists.
  siblings    sib_score, sib_name_tset, sib_addr_tset, sib_num_jacc, sib_num_first_eq:
              the candidate compared with the best OTHER candidate of the same S1.
              Copies of one business resemble each other (group consistency).

val/train features are built from train_all_candidates.parquet (every train S1
blocked) when it exists, so o_* competition features see all claimants, as on test.

Word rarity (IDF) is counted per country over all three sources of the split, one
file at a time; text is kept only for records that appear in candidate pairs, so
this fits in memory next to other jobs.

Run:  python src/features.py                  val + train + test
      python src/features.py --splits val     one split
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

import config
from blocking import _add_counts, _col, _fields, _lookup, cand_path
from io_utils import ID_BASE, load_gt_pairs
from normalize import IN_STATES, US_STATES, norm_path

TEXT_COLS = ["id", "country", "name_core", "name_alt", "name_nospace", "name_phon",
             "addr_clean", "addr_numbers", "has_indic"]
STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values())
CHUNK_ROWS = 1_000_000


def feat_path(name, cache_dir=config.CACHE_DIR):
    """Path of a saved feature file (name = val, train or test)."""
    return Path(cache_dir) / f"{name}_features.parquet"


def load_features(name, cache_dir=config.CACHE_DIR):
    """Load saved pair features, e.g. load_features("val")."""
    return pd.read_parquet(feat_path(name, cache_dir))


# --------------------------------------------------------------------------- text + word rarity

class _Side:
    """Text and word-rarity token lists (CSR layout) of the records one side needs."""

    def __init__(self, ids, text, tokens):
        self.index = pd.Index(ids)
        self.text = text
        self.tokens = tokens          # field -> (offsets, keys, idf)

    def rows(self, ids):
        """Row positions of entity ids (all must be present)."""
        pos = self.index.get_indexer(ids)
        if (pos < 0).any():
            raise ValueError("candidate id missing from the text store")
        return pos

    def take(self, col, rows):
        """Values of one text column at the given rows, as a numpy array."""
        return self.text.column(col).take(pa.array(rows)).to_numpy(zero_copy_only=False)


class PairText:
    """Normalised text + IDF token lists for the records appearing in candidate pairs.

    Document frequencies are counted over ALL records of the split (so rarity is the
    same as blocking sees it), but only the needed records are kept in memory.
    """

    FIELDS = ("name", "addr")

    def __init__(self, split, id1s, id2s):
        t0 = time.time()
        need = {1: np.unique(id1s), 23: np.unique(id2s)}
        dfreq = {f: None for f in self.FIELDS}
        country_n = {}
        kept = {1: [], 23: []}

        for s in (1, 2, 3):
            t = pq.read_table(norm_path(split, s), columns=TEXT_COLS).combine_chunks()
            fields, _ = _fields(t)
            for f in self.FIELDS:
                dfreq[f] = _add_counts(dfreq[f], fields[f][1])
            for c, n in pd.Series(_col(t, "country").to_numpy(zero_copy_only=False)).value_counts().items():
                country_n[c] = country_n.get(c, 0) + int(n)

            side = 1 if s == 1 else 23
            ids = _col(t, "id").to_numpy()
            rows = np.flatnonzero(np.isin(ids, need[side]))
            new_pos = np.full(len(ids), -1, dtype=np.int64)
            new_pos[rows] = np.arange(len(rows))
            toks = {}
            for f in self.FIELDS:
                r, k = fields[f]
                keep = new_pos[r] >= 0
                toks[f] = (new_pos[r[keep]], k[keep])
            kept[side].append((t.take(pa.array(rows)), toks))
            del t, fields
            gc.collect()

        self.s1 = self._side(kept[1], dfreq, country_n)
        self.other = self._side(kept[23], dfreq, country_n)
        print(f"  [{split}] text store: {len(self.s1.index):,} S1 + {len(self.other.index):,} S2/S3 "
              f"records ({time.time() - t0:.0f}s)", flush=True)

    @staticmethod
    def _side(parts, dfreq, country_n):
        """Stack the kept parts of one side and turn token keys into (offsets, keys, idf)."""
        text = pa.concat_tables([p[0] for p in parts]).combine_chunks()
        country = _col(text, "country").to_numpy(zero_copy_only=False)
        n_country = pd.Series(country).map(country_n).to_numpy(dtype=np.float64)
        tokens, offset = {}, 0
        for f in PairText.FIELDS:
            rows, keys, offset = [], [], 0
            for tbl, toks in parts:
                r, k = toks[f]
                rows.append(r + offset)
                keys.append(k)
                offset += tbl.num_rows
            r, k = np.concatenate(rows), np.concatenate(keys)
            order = np.argsort(r, kind="stable")
            r, k = r[order], k[order]
            idf = np.log(n_country[r] / _lookup(dfreq[f], k)).astype(np.float32)
            offsets = np.searchsorted(r, np.arange(text.num_rows + 1))
            tokens[f] = (offsets, k, idf)
        return _Side(_col(text, "id").to_numpy(), text, tokens)


def _expand(tokens, rows):
    """Token lists of the given rows, flattened -> (pair index, key, idf)."""
    offsets, keys, idf = tokens
    start = offsets[rows]
    count = offsets[rows + 1] - start
    total = int(count.sum())
    pair = np.repeat(np.arange(len(rows)), count)
    pos = np.arange(total) - np.repeat(np.cumsum(count) - count, count) + np.repeat(start, count)
    return pair, keys[pos], idf[pos], count


def idf_overlap(tok1, tok2, rows1, rows2):
    """Rarity-weighted overlap of two token lists per pair.

    Returns (jaccard, max idf of a shared word, share of side-1 rarity not shared,
    word count side 1, word count side 2).
    """
    n = len(rows1)
    p1, k1, w1, c1 = _expand(tok1, rows1)
    p2, k2, w2, c2 = _expand(tok2, rows2)
    s1 = np.bincount(p1, weights=w1, minlength=n)
    s2 = np.bincount(p2, weights=w2, minlength=n)
    p, k, w = np.concatenate([p1, p2]), np.concatenate([k1, k2]), np.concatenate([w1, w2])
    order = np.lexsort((k, p))
    p, k, w = p[order], k[order], w[order]
    dup = np.flatnonzero((p[1:] == p[:-1]) & (k[1:] == k[:-1])) + 1      # words present on both sides
    shared = np.bincount(p[dup], weights=w[dup], minlength=n)
    shared_max = np.zeros(n)
    np.maximum.at(shared_max, p[dup], w[dup])
    union = s1 + s2 - shared
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = np.where(union > 0, shared / union, 0.0)
        missing = np.where(s1 > 0, (s1 - shared) / s1, 1.0)
    return jacc, shared_max, missing, c1, c2


def _last_code(tokens):
    """State code of an address: its last token if it is a code, else its first, else None."""
    if tokens and tokens[-1] in STATE_CODES:
        return tokens[-1]
    if tokens and tokens[0] in STATE_CODES:
        return tokens[0]
    return None


def _cmp(a, b):
    """1 if both known and equal, 0 if both known and different, -1 if either unknown."""
    return -1 if a is None or b is None else int(a == b)


def number_place_features(addr1, addr2, nums1, nums2):
    """House number, number overlap, postcode and state agreement per pair (Python loop)."""
    n = len(addr1)
    out = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        a, b = nums1[i].split(), nums2[i].split()
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        pa_ = next((x for x in reversed(a) if len(x) in (5, 6)), None)
        pb_ = next((x for x in reversed(b) if len(x) in (5, 6)), None)
        out[i] = (inter / union if union else 0.0,
                  _cmp(a[0] if a else None, b[0] if b else None),
                  float(inter > 0), len(a), len(b),
                  _cmp(pa_, pb_),
                  _cmp(_last_code(addr1[i].split()), _last_code(addr2[i].split())))
    return out


# --------------------------------------------------------------------------- per-chunk features

def other_context(name):
    """Per S2/S3 record over the whole candidate file: claims count, best and 2nd-best score."""
    d = pd.read_parquet(cand_path(name), columns=["id2", "block_score"])
    d = d.sort_values(["id2", "block_score"], ascending=[True, False], kind="stable")
    pos = d.groupby("id2", sort=False).cumcount().to_numpy()       # 0 = best claim, 1 = second
    ctx = pd.DataFrame({"n": d.groupby("id2", sort=True).size()})
    ctx["top1"] = d.loc[pos == 0].set_index("id2")["block_score"].reindex(ctx.index)
    ctx["top2"] = d.loc[pos == 1].set_index("id2")["block_score"].reindex(ctx.index).fillna(0.0)
    return ctx


def sibling_features(pt, c, o, rank, score):
    """Similarity of each candidate to the best other candidate of the same S1 record."""
    sim = lambda a, b, scorer: process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)
    ids = c["id1"].to_numpy()
    first = pd.Series(o[rank == 0], index=ids[rank == 0])
    second = pd.Series(o[rank == 1], index=ids[rank == 1])
    s_first = pd.Series(score[rank == 0], index=ids[rank == 0])
    s_second = pd.Series(score[rank == 1], index=ids[rank == 1])
    is_top = rank == 0
    sib = np.where(is_top, second.reindex(ids).to_numpy(), first.reindex(ids).to_numpy())
    sib_score = np.where(is_top, s_second.reindex(ids).to_numpy(), s_first.reindex(ids).to_numpy())
    has = ~np.isnan(sib)                        # S1 records with a single candidate have no sibling
    sib = np.where(has, sib, 0).astype(np.int64)

    out = {"sib_score": np.where(has, sib_score, -1.0)}
    name_o, name_s = pt.other.take("name_core", o), pt.other.take("name_core", sib)
    addr_o, addr_s = pt.other.take("addr_clean", o), pt.other.take("addr_clean", sib)
    out["sib_name_tset"] = np.where(has, sim(name_o, name_s, fuzz.token_set_ratio), -1.0)
    out["sib_addr_tset"] = np.where(has, sim(addr_o, addr_s, fuzz.token_set_ratio), -1.0)
    num = number_place_features(addr_o, addr_s, pt.other.take("addr_numbers", o),
                                pt.other.take("addr_numbers", sib))
    out["sib_num_jacc"] = np.where(has, num[:, 0], -1.0)
    out["sib_num_first_eq"] = np.where(has, num[:, 1], -2.0)
    return out


def chunk_features(pt, c, octx):
    """All new features for one chunk of candidate pairs (whole S1 groups only)."""
    q, o = pt.s1.rows(c["id1"].to_numpy()), pt.other.rows(c["id2"].to_numpy())
    sim = lambda a, b, scorer: process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)
    F = {}

    core1, core2 = pt.s1.take("name_core", q), pt.other.take("name_core", o)
    ns1, ns2 = pt.s1.take("name_nospace", q), pt.other.take("name_nospace", o)
    F["name_ratio"] = sim(core1, core2, fuzz.ratio)
    F["name_partial_ns"] = sim(ns1, ns2, fuzz.partial_ratio)
    F["name_jw_ns"] = 100 * sim(ns1, ns2, JaroWinkler.normalized_similarity)
    F["phon_tset"] = sim(pt.s1.take("name_phon", q), pt.other.take("name_phon", o), fuzz.token_set_ratio)
    F["len1"] = np.array([len(x) for x in core1], dtype=np.float32)
    F["len2"] = np.array([len(x) for x in core2], dtype=np.float32)
    del core1, core2, ns1, ns2

    for f, prefix in (("name", "name"), ("addr", "addr")):
        jacc, mx, miss, n1, n2 = idf_overlap(pt.s1.tokens[f], pt.other.tokens[f], q, o)
        F[f"{prefix}_idf_jacc"], F[f"{prefix}_idf_max"] = jacc, mx
        if f == "name":
            F["name_idf_s1_missing"], F["name_n1"], F["name_n2"] = miss, n1, n2

    num = number_place_features(pt.s1.take("addr_clean", q), pt.other.take("addr_clean", o),
                                pt.s1.take("addr_numbers", q), pt.other.take("addr_numbers", o))
    for j, col in enumerate(["num_jacc", "num_first_eq", "num_any_eq", "num_n1", "num_n2",
                             "postcode_eq", "state_eq"]):
        F[col] = num[:, j]

    F["is_s3"] = (c["id2"].to_numpy() // ID_BASE == 3)
    F["has_indic1"] = pt.s1.take("has_indic", q)
    F["has_indic2"] = pt.other.take("has_indic", o)
    F["alt_nonempty"] = pt.other.take("name_alt", o).astype(bool)

    # context within this S1 record's candidates (whole groups are in the chunk)
    # (blocking's `rank` orders each S1 record's candidates by block_score, 0 = best)
    score = c["block_score"].to_numpy()
    rank = c["rank"].to_numpy()
    top1 = c["id1"].map(c.loc[rank == 0].set_index("id1")["block_score"]).to_numpy()
    top2 = c["id1"].map(c.loc[rank == 1].set_index("id1")["block_score"]).fillna(0.0).to_numpy()
    F["g_n_cand"] = c.groupby("id1", sort=False)["id2"].transform("size").to_numpy()
    F["g_n_strong"] = c.assign(s=score > 0.5).groupby("id1", sort=False)["s"].transform("sum").to_numpy()
    F["g_margin"] = score - np.where(rank == 0, top2, top1)

    # context across all S1 records claiming this S2/S3 record (whole candidate file)
    oc = octx.reindex(c["id2"].to_numpy())
    F["o_n_claims"] = oc["n"].to_numpy()
    F["o_margin"] = score - np.where(score >= oc["top1"].to_numpy(), oc["top2"].to_numpy(), oc["top1"].to_numpy())

    # group consistency: compare each candidate with its "sibling" = the best OTHER candidate
    # of the same S1 (rank 0, or rank 1 for the rank-0 candidate itself). Noisy copies of
    # one business resemble each other, so an unsure candidate that looks like a confident
    # sibling is probably a true match too.
    F.update(sibling_features(pt, c, o, rank, score))

    out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()})
    return pd.concat([c.reset_index(drop=True), out], axis=1)


def _group_chunks(source, ids=None, rows=CHUNK_ROWS):
    """Read candidates in chunks that never split one S1 record's candidates.

    ids restricts to some S1 records (read with a filter; rows keep file order, so each
    S1 record's candidates stay contiguous).
    """
    if ids is not None:
        df = pd.read_parquet(cand_path(source), filters=[("id1", "in", ids.tolist())])
        group_starts = np.flatnonzero(df["id1"].to_numpy()[1:] != df["id1"].to_numpy()[:-1]) + 1
        start = 0
        for g in group_starts:                  # cut only where a new S1 record begins
            if g - start >= rows:
                yield df.iloc[start:g]
                start = g
        yield df.iloc[start:]
        return
    pf = pq.ParquetFile(cand_path(source))
    carry = None
    for batch in pf.iter_batches(batch_size=rows):
        df = batch.to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        last = df["id1"].iat[-1]
        tail = df["id1"].to_numpy() == last
        carry = df[tail]
        if (~tail).any():
            yield df[~tail]
    if carry is not None and len(carry):
        yield carry


def source_of(name):
    """Candidate file a feature set is built from.

    val/train come from train_all_candidates (every train S1 blocked, so competition for
    an S2/S3 record is complete, as on test) when it exists, else from their own file.
    """
    if name in ("val", "train") and cand_path("train_all").exists():
        return "train_all"
    return name


# --------------------------------------------------------------------------- runs

def build(name, split):
    """Compute and save features for one set of candidate pairs; returns the output path."""
    from split import load_split

    t0 = time.time()
    source = source_of(name)
    s1_ids = load_split(name) if name in ("val", "train") and source == "train_all" else None
    filters = [("id1", "in", s1_ids.tolist())] if s1_ids is not None else None
    ids = pd.read_parquet(cand_path(source), columns=["id1", "id2"], filters=filters)
    pt = PairText(split, ids["id1"].to_numpy(), ids["id2"].to_numpy())
    n_pairs = len(ids)
    del ids
    octx = other_context(source)
    print(f"  [{name}] candidates from {cand_path(source).name}", flush=True)

    gt = None
    if split == "train":
        gt = load_gt_pairs()
        gt = gt.assign(label=np.int8(1))

    writer, done = None, 0
    try:
        for c in _group_chunks(source, s1_ids):
            f = chunk_features(pt, c, octx)
            if gt is not None:
                f = f.merge(gt, on=["id1", "id2"], how="left")
                f["label"] = f["label"].fillna(0).astype(np.int8)
            table = pa.Table.from_pandas(f, preserve_index=False)
            writer = writer or pq.ParquetWriter(feat_path(name), table.schema)
            writer.write_table(table.cast(writer.schema))
            done += len(f)
            print(f"  [{name}] {done:,}/{n_pairs:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    finally:
        if writer:
            writer.close()
    return feat_path(name)


def report(name="val"):
    """How well each feature alone separates true from false pairs (AUC), plus sanity checks."""
    from sklearn.metrics import roc_auc_score

    f = load_features(name)
    y = f["label"].to_numpy()
    skip = {"id1", "id2", "label"}
    rows = []
    for col in f.columns:
        if col in skip:
            continue
        x = f[col].to_numpy(dtype=np.float64)
        auc = roc_auc_score(y, np.nan_to_num(x)) if np.nanstd(x) > 0 else 0.5
        rows.append((col, max(auc, 1 - auc), np.isnan(x).mean(), np.nanmean(x[y == 1]), np.nanmean(x[y == 0])))
    r = pd.DataFrame(rows, columns=["feature", "auc", "nan_share", "mean_true", "mean_false"])
    r = r.sort_values("auc", ascending=False)
    print(f"\n[{name}] {len(f):,} pairs, {y.mean():.2%} true. Single-feature AUC (1.0 = separates perfectly):")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    constant = r.loc[r["auc"] == 0.5, "feature"].tolist()
    if constant:
        print(f"  constant features: {constant}")


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Pair features for the matching model")
    ap.add_argument("--splits", nargs="+", default=["val", "train", "test"],
                    choices=["val", "train", "test", "train_all"],
                    help="train_all = every train S1 record (needed by stage2.py)")
    args = ap.parse_args(argv)
    for name in args.splits:
        build(name, "test" if name == "test" else "train")
        gc.collect()
        if name == "val":
            report("val")


if __name__ == "__main__":
    sys.exit(main())
