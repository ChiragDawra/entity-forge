"""Deterministic record normalization.

All heavy lifting is vectorized with Polars expressions. The only per-string
Python work is transliteration, applied to the *unique* non-ASCII values.

Output columns:

- ``ckey``        lowercase country label (open set, never filtered)
- ``name_norm``   transliterated, lowercased, punctuation-free business name
- ``name_core``   ``name_norm`` without legal-form and filler tokens
- ``name_alt``    text after a DBA / "formerly" marker, else ``name_core``
- ``name_phon``   consonant skeleton of ``name_core`` (cross-script matching)
- ``addr_norm``   canonical address tokens (abbreviations, regions, ordinals)
- ``addr_nums``   unique digit runs of the address, leading zeros stripped
- ``has_addr``    address present after cleanup
- ``translit``    name contained non-Latin script
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl
from anyascii import anyascii

from . import dictionaries as D

# Nasal-sign codepoints across Indic scripts. anyascii renders them as "m",
# which turns "marketing" into "marketimg"; "n" is the better default.
_ANUSVARA = re.compile(
    "[ँंঁংਁਂੰઁંଁଂ"
    "ఁంಁಂഁം]"
)
_NON_ASCII = r"[^\x00-\x7F]"
_NON_LATIN = r"[^\x00-\x{024F}]"


def transliterate(text: str) -> str:
    """Return an ASCII rendering of ``text`` (accents folded, scripts romanized)."""
    if text.isascii():
        return text
    text = unicodedata.normalize("NFKC", text)
    text = _ANUSVARA.sub("n", text)
    return anyascii(text)


def _transliterate_column(col: pl.Series) -> pl.Series:
    """Transliterate only the unique non-ASCII values of ``col``."""
    mask = col.str.contains(_NON_ASCII)
    if not mask.any():
        return col
    uniq = col.filter(mask).unique().to_list()
    mapping = {s: transliterate(s) for s in uniq}
    return col.replace(mapping)


def _collapse(expr: pl.Expr) -> pl.Expr:
    return expr.str.replace_all(r"\s+", " ").str.strip_chars()


def _token_map(tokens: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    return tokens.list.eval(pl.element().replace(mapping))


def _drop_tokens(tokens: pl.Expr, drop: frozenset[str]) -> pl.Expr:
    return tokens.list.eval(pl.element().filter(~pl.element().is_in(sorted(drop))))


# Digit look-alikes inside alphabetic tokens ("5arasaksh", "l0gistics").
_LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}


def _fix_leet(tokens: pl.Expr) -> pl.Expr:
    fixed = pl.element()
    for digit, letter in _LEET.items():
        fixed = fixed.str.replace_all(digit, letter, literal=True)
    looks_leet = (
        pl.element().str.contains(r"^[a-z]*[0134578][a-z]*$")
        & (pl.element().str.count_matches(r"[a-z]") >= 3)
    )
    return tokens.list.eval(pl.when(looks_leet).then(fixed).otherwise(pl.element()))


# Ordered rewrite rules: English spelling and anyascii's schwa-less romanization
# of Indic names converge ("life finance" / "laiph phainens" -> "lf fnns").
_PHONETIC_REGEX: tuple[tuple[str, str], ...] = (
    (r"tion", "shn"),
    (r"c([eiy])", "s$1"),  # soft c
    (r"g([eiy])", "j$1"),  # soft g
    (r"j\b", "s"),  # Devanagari plural "-ज" for English "-s"
)
_PHONETIC_STEPS: tuple[tuple[str, str], ...] = (
    ("ph", "f"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
    ("gh", "g"), ("jh", "j"), ("wh", "v"), ("vh", "v"), ("ch", "c"), ("ck", "k"), ("c", "k"),
    ("q", "k"), ("x", "ks"), ("z", "s"), ("w", "v"), ("y", "i"), ("'", ""),
    ("d", "t"),  # Devanagari retroflex d is used for English t
)


def phonetic_expr(expr: pl.Expr) -> pl.Expr:
    """Consonant skeleton: vowels dropped after the first letter, digraphs folded.

    Makes anyascii's schwa-less romanization comparable with the English
    spelling: "mharastr" / "maharashtra" -> "mhrstr", "laiph phainens" /
    "life finance" -> "lf fns".
    """
    out = expr
    for pattern, repl in _PHONETIC_REGEX:
        out = out.str.replace_all(pattern, repl)
    for src, dst in _PHONETIC_STEPS:
        out = out.str.replace_all(src, dst, literal=True)
    out = out.str.replace_all(r"\B[aeiou]", "")
    out = out.str.replace_all(r"\b[aeiou]", "a")  # initial vowels are unreliable too
    for ch in "bfgjklmnprstv":
        out = out.str.replace_all(f"{ch}{ch}+", ch)
    return _collapse(out)


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------

_DBA_RE = r"\b(?:" + "|".join(re.escape(m) for m in D.DBA_MARKERS) + r")\b"
_NAME_STOP = D.LEGAL_TOKENS | D.FILLER_TOKENS


def _normalize_names(df: pl.DataFrame) -> pl.DataFrame:
    name = pl.col("_name").str.to_lowercase()
    name = name.str.replace_all(r"\b([a-z])\.", "$1")  # "p.c." -> "pc", "l.l.c." -> "llc"
    name = name.str.replace_all(r"\|?\s*(?:https?://)?www\.[^\s|]+", " ")
    name = name.str.replace_all(r"\.(?:com|in|net|org|co|fr|us|biz|info)\b", " ")
    name = name.str.replace_all("&", " and ", literal=True)
    name = name.str.replace_all(r"[^a-z0-9 ]", " ")
    tokens = _token_map(_fix_leet(_collapse(name).str.split(" ")), D.NAME_TOKEN_MAP)
    df = df.with_columns(tokens.list.join(" ").alias("_nn"))

    has_dba = pl.col("_nn").str.contains(_DBA_RE)
    df = df.with_columns(
        pl.when(has_dba)
        .then(_collapse(pl.col("_nn").str.replace(r"^.*" + _DBA_RE, "")))
        .otherwise(pl.lit(""))
        .alias("_alt"),
        _collapse(pl.col("_nn").str.replace_all(_DBA_RE, " ")).alias("name_norm"),
    )

    core = _drop_tokens(pl.col("name_norm").str.split(" "), _NAME_STOP).list.join(" ")
    fallback = _drop_tokens(pl.col("name_norm").str.split(" "), D.FILLER_TOKENS).list.join(" ")
    df = df.with_columns(
        pl.when(core.str.len_chars() > 0)
        .then(core)
        .when(fallback.str.len_chars() > 0)
        .then(fallback)
        .otherwise(pl.col("name_norm"))
        .alias("name_core")
    )
    alt_core = _drop_tokens(pl.col("_alt").str.split(" "), _NAME_STOP).list.join(" ")
    df = df.with_columns(
        pl.when(alt_core.str.len_chars() > 0).then(alt_core).otherwise(pl.col("name_core")).alias("name_alt"),
        phonetic_expr(pl.col("name_core")).alias("name_phon"),
    )
    return df.drop("_nn", "_alt")


# ---------------------------------------------------------------------------
# Address
# ---------------------------------------------------------------------------


def _replace_phrases(expr: pl.Expr, table: dict[str, str]) -> pl.Expr:
    """Replace multi-word phrases (longest first) with their codes."""
    for phrase in sorted((p for p in table if " " in p), key=len, reverse=True):
        expr = expr.str.replace_all(r"\b" + re.escape(phrase) + r"\b", table[phrase])
    return expr


def _normalize_address_partition(df: pl.DataFrame, ckey: str) -> pl.DataFrame:
    regions = D.REGION_TABLES.get(ckey, {})
    token_map = dict(D.ADDRESS_TOKEN_MAPS.get(ckey, D.ADDRESS_TOKEN_MAP_GENERIC))
    token_map.update({k: v for k, v in regions.items() if " " not in k})

    addr = pl.col("_addr").str.to_lowercase()
    addr = addr.str.replace_all(r"(\d+)\s*(?:st|nd|rd|th)\b", "$1")
    addr = addr.str.replace_all(r"[^a-z0-9 ]", " ")
    addr = addr.str.replace_all(r"\b0+(\d)", "$1")
    addr = _replace_phrases(_collapse(addr), regions)
    toks = _token_map(_collapse(addr).str.split(" "), token_map)
    toks = _drop_tokens(toks, D.ADDRESS_DROP_TOKENS)
    toks = toks.list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
    return df.with_columns(toks.list.unique(maintain_order=True).list.join(" ").alias("addr_norm"))


def _replace_indic_states(col: pl.Series) -> pl.Series:
    if not col.str.contains(_NON_ASCII).any():
        return col
    for native, code in sorted(D.INDIC_STATE_NAMES.items(), key=lambda kv: -len(kv[0])):
        col = col.str.replace_all(native, f" {code} ", literal=True)
    return col


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS: tuple[str, ...] = (
    "entity_id", "country", "ckey", "business_name", "business_address", "name_norm",
    "name_core", "name_alt", "name_phon", "addr_norm", "addr_nums", "has_addr", "translit",
)


def normalize_records(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize a raw source frame (entity_id, business_name, business_address, country)."""
    df = df.with_columns(
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    )
    df = df.with_columns(
        pl.col("country").str.strip_chars().str.to_lowercase().alias("ckey"),
        pl.col("business_name").str.contains(_NON_LATIN).alias("translit"),
    )
    df = df.with_columns(
        _transliterate_column(df["business_name"]).alias("_name"),
        _transliterate_column(_replace_indic_states(df["business_address"])).alias("_addr"),
    )
    df = _normalize_names(df)

    parts = [
        _normalize_address_partition(part, str(ckey))
        for (ckey,), part in df.group_by(["ckey"], maintain_order=True)
    ]
    df = pl.concat(parts) if parts else df.with_columns(pl.lit("").alias("addr_norm"))

    df = df.with_columns(
        pl.col("addr_norm")
        .str.extract_all(r"\d+")
        .list.eval(pl.element().str.strip_chars_start("0"))
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        .list.unique(maintain_order=True)
        .alias("addr_nums"),
        (pl.col("addr_norm").str.len_chars() > 0).alias("has_addr"),
    )
    return df.select(OUTPUT_COLUMNS)
