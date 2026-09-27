"""Clean business names and addresses so different spellings of one business look alike.

Every later stage (blocking, features) works on the output of this file, never on the
raw text. The rules come from the noise actually seen in the training data:

  names     Indian scripts (7 of them), accents, junk punctuation ([ ] ( ) * -- << |),
            "(ID: 12345)" tags, web domains used as names, titles (Shri, Smt, Dr, M/s),
            area words added to names (Midtown, Northside...), aliases (dba, formerly,
            fka, aka), look-alike characters (Appare1s, L0ndon, lnc), legal forms in
            English, Indian and French variants (LLC, Pvt Ltd, SARL...).
  addresses NULL / <NULL> placeholders, landmarks (near X, opp. Y), abbreviations
            (St/Street/Saint, Rd/Road, R/Rue...), state names vs codes (Texas/TX,
            Maharashtra/MH, native-script state names), "CITY" suffixes, extra leading
            zeros on numbers (02120 -> 2120), old city names (Bombay -> mumbai).

Nothing here filters or branches on country, so France (test only) goes through the
same generic rules; French legal forms and street abbreviations are simply included.

Output per record (cached as <cache>/{split}_source{n}_norm.parquet):
    id, country,
    name_clean    all name tokens after cleaning
    name_core     name_clean minus legal words, titles, area words, stop words
    name_alt      core of an alias part ("X dba Y" -> Y), else ""
    name_nospace  name_core without spaces (matches domain-style names)
    name_phon     sound key of each name_core word (matches rough transliterations)
    addr_clean    cleaned address tokens
    addr_numbers  numbers in the address, leading zeros removed, in order of appearance
    has_indic     the raw name or address contained an Indian script

Run:  python src/normalize.py            build the cache (skips files already built), then report
      python src/normalize.py --check    only the before/after report on the val split
      python src/normalize.py --force    rebuild everything
"""
import argparse
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from anyascii import anyascii

import config
from io_utils import SOURCES, SPLITS, load_gt_pairs, load_source

# --------------------------------------------------------------------------- word lists

LEGAL = {
    # English / US
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "lp", "llp", "pc", "pllc", "plc",
    # India
    "pvt", "private", "public",
    # France
    "sarl", "sas", "sasu", "sa", "sci", "ei", "eurl", "snc", "compagnie", "cie",
    # truncated forms seen in the data (Prívate -> vate, Limited -> mited)
    "vate", "mited",
}
TITLES = {"shri", "shree", "smt", "sri", "mr", "mrs", "ms", "dr", "messrs"}   # "M/s" -> "ms"
AREA = {"midtown", "northside", "southside", "eastgate", "westgate", "lakeside", "riverside", "greater"}
STOP = {"the", "and", "of", "a", "an"}
WEB = {"com", "www"}          # left over from domains written without dots ("eastking com")
NAME_DROP = LEGAL | TITLES | AREA | STOP | WEB

# Common business words inside Indian-script names, after anyascii transliteration.
# Taken from the most frequent transliterated tokens in the train data. Applied only to
# text that was written in an Indian script, so English names like "Li" are untouched.
INDIC_WORDS = {
    "praivet": "private", "praibhet": "private", "piraivet": "private", "praivrr": "private",
    "pra": "private", "limitet": "limited", "limirrd": "limited", "li": "limited",
    "elelpi": "llp", "tredimg": "trading", "imtrnesnl": "international", "impeks": "impex",
    "enrji": "energy", "phuds": "foods", "phud": "food", "prodkts": "products",
    "prodyusr": "producer", "bildrs": "builders", "sistms": "systems", "devlprs": "developers",
    "eksports": "exports", "imjiniyrimg": "engineering", "midiya": "media",
    "marketimg": "marketing", "phaumdesn": "foundation", "vemcrs": "ventures",
    "imdstrij": "industries", "solyusms": "solutions", "kmstrksms": "constructions",
    "kmstrksn": "construction", "emtrpraijej": "enterprises", "teknolojij": "technologies",
    "teknoloji": "technology", "bijnes": "business", "egro": "agro", "keyr": "care",
    "projekts": "projects", "lojistiks": "logistics", "kmsltimg": "consulting",
    "imphrastrkcr": "infrastructure", "hospitailiti": "hospitality",
    "mainejmemt": "management", "imphotek": "infotech", "srvisej": "services",
    "investmemts": "investments", "proprtij": "properties", "sophtveyr": "software",
    "kmsltemts": "consultants", "kmsltemsi": "consultancy", "globl": "global",
    "estet": "estate", "tek": "tech", "aiti": "it", "pavr": "power", "phainems": "finance",
    "imphra": "infra",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {   # codes as used in the data (vehicle-registration style: MH, DL, TG...)
    "maharashtra": "mh", "delhi": "dl", "uttar pradesh": "up", "karnataka": "ka",
    "tamil nadu": "tn", "west bengal": "wb", "gujarat": "gj", "telangana": "tg",
    "haryana": "hr", "rajasthan": "rj", "kerala": "kl", "bihar": "br",
    "madhya pradesh": "mp", "andhra pradesh": "ap", "punjab": "pb", "odisha": "od",
    "orissa": "od", "jharkhand": "jh", "chhattisgarh": "cg", "assam": "as",
    "uttarakhand": "uk", "himachal pradesh": "hp", "goa": "ga", "jammu and kashmir": "jk",
}
# Native-script state names in addresses, after anyascii (all 16 seen in the data).
INDIC_STATES = {
    "mharastr": "mh", "dilli": "dl", "uttr prdes": "up", "krnatk": "ka", "tmilnatu": "tn",
    "pscimbng": "wb", "gujrat": "gj", "telmgan": "tg", "hriyana": "hr", "rajsthan": "rj",
    "kerlm": "kl", "bihar": "br", "mdhy prdes": "mp", "amdhrprdes": "ap", "pmjab": "pb",
    "od'isa": "od", "odisa": "od",
}
PLACE_ALIASES = {"bengaluru": "bangalore", "bombay": "mumbai", "gurgaon": "gurugram",
                 "calcutta": "kolkata", "madras": "chennai"}
STATE_PHRASES = {**US_STATES, **IN_STATES, **PLACE_ALIASES}

# Every variant maps to ONE short form, so e.g. Street / St / Saint all agree.
ADDR_MAP = {
    "street": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir", "place": "pl",
    "boulevard": "blvd", "bd": "blvd", "highway": "hwy", "parkway": "pkwy",
    "suite": "ste", "apartment": "apt", "floor": "fl", "flr": "fl", "building": "bldg",
    "rue": "r", "route": "rte", "north": "n", "south": "s", "east": "e", "west": "w",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
    "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th",
}
ADDR_DROP = {"city", "no", "number", "null"}

# --------------------------------------------------------------------------- regexes

_INDIC = re.compile(r"[ऀ-෿][ऀ-෿\s]*")          # Devanagari ... Malayalam
_ID_TAG = re.compile(r"\(?\bid\s*[:#]\s*\d+\s*\)?|#\s*\d+")   # "(ID: 11481)", "#68886"
_ALIAS = re.compile(r"\b(?:d[./]?b[./]?a|f[./]?k[./]?a|a[./]?k[./]?a|formerly known as|formerly|"
                    r"doing business as|also known as|known as|trading as)\b\.?")
_WWW = re.compile(r"\bwww\.")
_DOMAIN = re.compile(r"\b([a-z0-9-]+)\.(?:co\.in|com|c0m|net|org|biz|info|in|fr|us|io|co)\b")
_AMP = re.compile(r"[&+]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NULL = re.compile(r"<\s*null\s*>|\bnull\b|\bn\s*/\s*a\b")
_LANDMARK = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to)\b[^,]*")
_STATE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATE_PHRASES), key=len, reverse=True)) + r")\b")
_DIGITS = re.compile(r"\d+")
_LOOKALIKE = str.maketrans({"0": "o", "1": "l"})

# Sound key: spellings that sound alike reduce to the same consonant skeleton.
_PHON_PAIRS = [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
               ("sh", "s"), ("ck", "k"), ("ch", "k"), ("c", "k"), ("q", "k"), ("x", "ks"),
               ("v", "w"), ("z", "j")]
_M_BEFORE_CONS = re.compile(r"m(?=[bdfgjkpstw])")   # Indian-script nasal: "imdstrij" ~ "industries"
_VOWELS = re.compile(r"[aeiouyh]")
_REPEATS = re.compile(r"(.)\1+")


# --------------------------------------------------------------------------- cleaning

def to_ascii(text, indic_map):
    """Lowercase ASCII version of text; Indian-script runs are transliterated and then
    fixed with indic_map (whole phrase first, then word by word)."""
    def fix(m):
        words = anyascii(m.group()).lower().split()
        phrase = " ".join(words)
        if phrase in indic_map:
            return f" {indic_map[phrase]} "
        return " " + " ".join(indic_map.get(w, w) for w in words) + " "
    return anyascii(_INDIC.sub(fix, text)).lower()


def _merge_initials(tokens):
    """Join runs of single letters: 't k elite' -> 'tk elite', 'l l c' -> 'llc'."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def _fix_lookalike(tok):
    """Undo digit-for-letter swaps inside mostly-letter words: appare1s -> apparels."""
    if tok == "lnc":
        return "inc"
    n_digit = sum(c.isdigit() for c in tok)
    if 0 < n_digit < len(tok) - n_digit:
        return tok.translate(_LOOKALIKE)
    return tok


def _name_tokens(text):
    """Tokenise one (already lowercase ASCII) name part."""
    text = _WWW.sub(" ", text)
    text = _DOMAIN.sub(r" \1 ", text)
    text = text.replace("'", "").replace("`", "")
    text = _AMP.sub(" and ", text)
    tokens = _merge_initials(_NON_ALNUM.sub(" ", text).split())
    return [_fix_lookalike(t) for t in tokens]


def phonetic(token):
    """Consonant skeleton of one word, tolerant to transliteration and vowel changes.

    'bharat' / 'bhart' -> 'brt', 'software' / 'sophtver' -> 'sftwr', 'apex' / 'apeks' -> 'apks'.
    """
    if not token.isalpha():
        return token
    for a, b in _PHON_PAIRS:
        token = token.replace(a, b)
    token = _M_BEFORE_CONS.sub("n", token)
    return _REPEATS.sub(r"\1", token[0] + _VOWELS.sub("", token[1:]))


def _core(tokens):
    """Drop legal words, titles, area words and stop words; keep the rest if that empties it."""
    core = [t for t in tokens if t not in NAME_DROP]
    return core or [t for t in tokens if t not in STOP]


def clean_name(raw):
    """Return (name_clean, name_core, name_alt, name_nospace, name_phon) for one raw name."""
    text = _ID_TAG.sub(" ", to_ascii(raw, INDIC_WORDS))

    # "X LLC | www.x.com": keep the first part that is not just a web address
    parts = [p.strip() for p in text.split("|") if p.strip()] or [""]
    text = next((p for p in parts if not (_WWW.search(p) or _DOMAIN.fullmatch(p))), parts[0])

    # "X dba Y", "X formerly Y": X is the name, Y the alias (swap if X is empty)
    main, alt = _split_alias(text)
    tokens = _name_tokens(main)
    alt_tokens = _name_tokens(alt)
    if not tokens:
        tokens, alt_tokens = alt_tokens, []
    # a long single word ending in "com" is a domain written without its dot
    if len(tokens) == 1 and len(tokens[0]) >= 12 and tokens[0].endswith("com"):
        tokens = [tokens[0][:-3]]

    core = _core(tokens)
    alt_core = _core(alt_tokens) if alt_tokens else []
    phon = " ".join(phonetic(t) for t in core)
    return " ".join(tokens), " ".join(core), " ".join(alt_core), "".join(core), phon


def _split_alias(text):
    """Split 'main <alias marker> alt' into (main, alt); alt is '' when there is no marker."""
    pieces = _ALIAS.split(text, maxsplit=1)
    return (pieces[0], pieces[1]) if len(pieces) == 2 else (text, "")


def clean_address(raw):
    """Return (addr_clean, addr_numbers) for one raw business address."""
    text = _NULL.sub(" ", to_ascii(raw, INDIC_STATES))
    text = _LANDMARK.sub(",", text)          # needs the commas, so before punctuation removal
    text = text.replace("'", "")
    text = " " + _NON_ALNUM.sub(" ", text) + " "
    text = _STATE.sub(lambda m: STATE_PHRASES[m.group(1)], text)

    tokens = []
    for t in text.split():
        if t in ADDR_DROP:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
        tokens.append(ADDR_MAP.get(t, t))

    numbers = []
    for n in _DIGITS.findall(" ".join(tokens)):
        n = n.lstrip("0") or "0"
        if n not in numbers:
            numbers.append(n)
    return " ".join(tokens), " ".join(numbers)


def normalize_chunk(args):
    """Clean one chunk of (names, addresses); returns a dict of output columns.

    Top-level so multiprocessing can send it to worker processes.
    """
    names, addrs = args
    cols = {k: [] for k in ("name_clean", "name_core", "name_alt", "name_nospace", "name_phon",
                            "addr_clean", "addr_numbers", "has_indic")}
    for name, addr in zip(names, addrs):
        nc, core, alt, nospace, phon = clean_name(name)
        ac, nums = clean_address(addr)
        cols["name_clean"].append(nc)
        cols["name_core"].append(core)
        cols["name_alt"].append(alt)
        cols["name_nospace"].append(nospace)
        cols["name_phon"].append(phon)
        cols["addr_clean"].append(ac)
        cols["addr_numbers"].append(nums)
        cols["has_indic"].append(bool(_INDIC.search(name) or _INDIC.search(addr)))
    return cols


# --------------------------------------------------------------------------- caching

def norm_path(split, source, cache_dir=config.CACHE_DIR):
    """Path of one normalised cache file."""
    return Path(cache_dir) / f"{split}_source{source}_norm.parquet"


def normalize_file(split, source, pool, chunk_rows=50_000, write_rows=500_000):
    """Normalise one cached source file in parallel and stream it to parquet."""
    df = load_source(split, source, columns=["id", "country", "business_name", "business_address"])
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    chunks = [(names[i:i + chunk_rows], addrs[i:i + chunk_rows]) for i in range(0, len(df), chunk_rows)]

    out, writer, start, buffered = norm_path(split, source), None, 0, []
    try:
        for cols in pool.imap(normalize_chunk, chunks):
            buffered.append(cols)
            n_buf = sum(len(c["name_core"]) for c in buffered)
            if n_buf >= write_rows or start + n_buf == len(df):
                block = df.iloc[start:start + n_buf]
                table = pa.table({
                    "id": block["id"].to_numpy(),
                    "country": block["country"].astype(str).to_numpy(),
                    **{k: sum((c[k] for c in buffered), []) for k in buffered[0]},
                })
                writer = writer or pq.ParquetWriter(out, table.schema)
                writer.write_table(table)
                start += n_buf
                buffered = []
    finally:
        if writer:
            writer.close()
    return len(df)


def load_norm(split, source, columns=None, cache_dir=config.CACHE_DIR):
    """Load one normalised file, e.g. load_norm("test", 2, columns=["id", "name_core"])."""
    return pd.read_parquet(norm_path(split, source, cache_dir), columns=columns)


def build_all(force=False):
    """Normalise all six source files (skipping ones already built unless force)."""
    todo = [(sp, s) for sp in SPLITS for s in SOURCES if force or not norm_path(sp, s).exists()]
    if not todo:
        print("Normalised cache already complete (use --force to rebuild)")
        return
    with Pool(config.N_WORKERS) as pool:
        for split, source in todo:
            t0 = time.time()
            n = normalize_file(split, source, pool)
            print(f"  {split}_source{source}: {n:,} rows in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- checks

def _self_test():
    """Examples taken from the data; fails loudly if a rule stops working."""
    assert clean_name("Lucknow Appare1s")[1] == "lucknow apparels"
    assert clean_name("A SOBTRNATMGIC ENVIRONMENTAL")[1] == "sobtrnatmgic environmental"
    assert clean_name("Espinoza & LLC Industrials Davis")[1] == "espinoza industrials davis"
    assert clean_name("Espinoza & Davis Industrials L.L.C.")[1] == "espinoza davis industrials"
    assert clean_name("Onyxveo formerly Espinoza & Davis Industrials LLC")[1:3] == \
        ("onyxveo", "espinoza davis industrials")
    assert clean_name("STRATEGIC BUSINESS  UNION INC (ID: 11481)")[1] == "strategic business union"
    assert clean_name("Casella's Highland Tattoo LLC | www.casellas.com")[1] == "casellas highland tattoo"
    assert clean_name("britconnollymitsui.com")[1] == "britconnollymitsui"
    assert clean_name("M/s Smt Darshan Communications Private Limited")[1] == "darshan communications"
    assert clean_name("Cripe Select Skin Care Lakeside Inc.")[1] == "cripe select skin care"
    assert clean_name("राम मार्केटिंग प्राइवेट लिमिटेड")[1] == "ram marketing"
    assert clean_name("RI Amicale SARL")[1] == "ri amicale"
    assert clean_name("CONTINENTALQUASAREDGECOM")[3] == clean_name("Continental Quasaredge")[3]
    assert clean_name("eastking.c0m")[3] == clean_name("East King LP")[3] == "eastking"
    assert clean_name("Shri Janashakti Square Private Limited Services #68886")[1] == \
        "janashakti square services"
    assert clean_name("హరి ఇన్ఫోటెక్ ప్రైవేట్ లిమిటెడ్")[4] == clean_name("Hari Infotech Private Limited")[4]
    assert clean_name("આનંદ સોફ્ટવેર પ્રાઇવેટ લિમિટેડ")[4] == clean_name("Anand Software Private Limited")[4]
    assert clean_name("அபெக்ஸ் கன்ஸ்ட்ரக்ஷன்ஸ் பிரைவேட் லிமிடெட்")[4].split()[0] == \
        clean_name("Apex Constructions Private Limited")[4].split()[0]
    assert clean_address("2502 Pixton Dr, N/A, Killeen, TX")[0] == "2502 pixton dr killeen tx"
    assert clean_address("2 MULDOWNEY CIR, POUGHKEEPSIE, NY") == ("2 muldowney cir poughkeepsie ny", "2")
    assert clean_address("2 Muldowney Circle, Poughkeepsie, New York")[0] == "2 muldowney cir poughkeepsie ny"
    assert clean_address("506 Boone Saint, Olney, Illinois")[0] == "506 boone st olney il"
    assert clean_address("3310 Anita St, NULL, Houston CITY, Texas")[0] == "3310 anita st houston tx"
    assert clean_address("02120 BREVARD STREET, CHARLOTTE, NC") == ("2120 brevard st charlotte nc", "2120")
    assert clean_address("4/792., SECTOR 4, VIKAS NAGAR, LUCKNOW, उत्तर प्रदेश")[0] == \
        "4 792 sector 4 vikas nagar lucknow up"
    assert clean_address("136, WORLD TRADE CENTRE, NEAR UDHANA DARWAJA, RING ROAD, SURAT CITY")[0] == \
        "136 world trade centre ring rd surat"
    assert clean_address("43 RUE DE MADRID, BORDEAUX, Gironde")[0] == "43 r de madrid bordeaux gironde"
    print("normalize self-test passed")


def _jaccard(a, b):
    """Token-set Jaccard similarity of two space-separated strings."""
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a and b else 0.0


def report(n_show=6):
    """Before/after comparison on the val split's true pairs vs random same-country pairs."""
    from split import load_split

    val = load_split("val")
    pairs = load_gt_pairs()
    pairs = pairs[pairs["id1"].isin(val)].reset_index(drop=True)
    need = set(pairs["id1"]) | set(pairs["id2"])

    raw = pd.concat([load_source("train", s, columns=["id", "business_name", "business_address"])
                     for s in SOURCES])
    norm = pd.concat([load_norm("train", s) for s in SOURCES])
    raw, norm = raw[raw["id"].isin(need)].set_index("id"), norm[norm["id"].isin(need)].set_index("id")

    # random pairs: same S1 records, partner shuffled among pairs of the same country
    country = norm["country"]
    rnd = pairs.copy()
    rng = np.random.default_rng(config.SEED)
    for c, idx in rnd.groupby(rnd["id1"].map(country)).groups.items():
        rnd.loc[idx, "id2"] = rng.permutation(rnd.loc[idx, "id2"].to_numpy())

    raw_tok = lambda s: " ".join(re.findall(r"\w+", s.lower()))
    raw_num = lambda s: " ".join(dict.fromkeys(re.findall(r"\d+", s)))

    def measure(p):
        r1, r2 = raw.loc[p["id1"]], raw.loc[p["id2"]]
        n1, n2 = norm.loc[p["id1"]], norm.loc[p["id2"]]
        rn1, rn2 = [raw_tok(x) for x in r1["business_name"]], [raw_tok(x) for x in r2["business_name"]]
        ra1, ra2 = [raw_tok(x) for x in r1["business_address"]], [raw_tok(x) for x in r2["business_address"]]
        best_core = [max(_jaccard(a, b), _jaccard(a, c) if c else 0.0)
                     for a, b, c in zip(n1["name_core"], n2["name_core"], n2["name_alt"])]
        return {
            "name exact match": (np.mean([a == b for a, b in zip(rn1, rn2)]),
                                 np.mean([a == b and a != "" for a, b in zip(n1["name_core"], n2["name_core"])])),
            "name word overlap": (np.mean([_jaccard(a, b) for a, b in zip(rn1, rn2)]), np.mean(best_core)),
            "address word overlap": (np.mean([_jaccard(a, b) for a, b in zip(ra1, ra2)]),
                                     np.mean([_jaccard(a, b) for a, b in zip(n1["addr_clean"], n2["addr_clean"])])),
            "address number overlap": (
                np.mean([_jaccard(raw_num(a), raw_num(b)) for a, b in zip(r1["business_address"], r2["business_address"])]),
                np.mean([_jaccard(a, b) for a, b in zip(n1["addr_numbers"], n2["addr_numbers"])])),
            "share a name word": (np.mean([bool(set(a.split()) & set(b.split())) for a, b in zip(rn1, rn2)]),
                                  np.mean([x > 0 for x in best_core])),
            # what blocking can use: a shared word, a shared sound word, or equal no-space names
            "+ sound word / nospace": (
                np.mean([bool(set(a.split()) & set(b.split())) for a, b in zip(rn1, rn2)]),
                np.mean([x > 0 or bool(set(pa.split()) & set(pb.split())) or (sa == sb and sa != "")
                         for x, pa, pb, sa, sb in zip(best_core, n1["name_phon"], n2["name_phon"],
                                                     n1["name_nospace"], n2["name_nospace"])])),
        }

    true_m, rnd_m = measure(pairs), measure(rnd)
    print(f"\nVal split: {len(pairs):,} true pairs vs {len(rnd):,} random same-country pairs")
    print(f"  {'':<24} {'true raw':>9} {'true clean':>11}   {'random raw':>10} {'random clean':>12}")
    for k in true_m:
        print(f"  {k:<24} {true_m[k][0]:>9.3f} {true_m[k][1]:>11.3f}   {rnd_m[k][0]:>10.3f} {rnd_m[k][1]:>12.3f}")

    print("\nExamples (raw -> name_core | addr_clean):")
    for id1 in pairs["id1"].drop_duplicates().head(n_show):
        for i in [id1] + pairs.loc[pairs["id1"] == id1, "id2"].tolist():
            r, n = raw.loc[i], norm.loc[i]
            print(f"  {r['business_name'][:38]:<38} -> {n['name_core'][:30]:<30} | {n['addr_clean'][:50]}")
        print()


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Normalise names and addresses")
    ap.add_argument("--check", action="store_true", help="only run the before/after report")
    ap.add_argument("--force", action="store_true", help="rebuild all normalised files")
    args = ap.parse_args(argv)

    _self_test()
    if not args.check:
        t0 = time.time()
        build_all(args.force)
        print(f"Built in {time.time() - t0:.0f}s")
    report()


if __name__ == "__main__":
    sys.exit(main())
