import polars as pl

from entity_forge.normalize import normalize_records, transliterate


def norm(rows):
    df = pl.DataFrame(
        rows,
        schema={"entity_id": pl.String, "business_name": pl.String,
                "business_address": pl.String, "country": pl.String},
        orient="row",
    )
    return {r["entity_id"]: r for r in normalize_records(df).iter_rows(named=True)}


def test_transliteration_handles_devanagari_and_accents():
    assert transliterate("मार्केटिंग") == "marketing"  # anusvara -> n, not m
    assert transliterate("Léarning") == "Learning"
    assert transliterate("plain ascii") == "plain ascii"


def test_legal_suffix_and_reordering_share_core():
    out = norm([
        ("S1-1", "Patterson Chadwick Corp", "1673 218th Avenue, Buckeye, AZ", "US"),
        ("S2-1", "PATTERSON CHADWICK (CORP)", "001673 218RD AVENUE, BUCKEYE, AZ", "US"),
        ("S3-1", "Patterson Corp (Chadwick)", "218rd Avenue, Buckeye, Arizona", "US"),
    ])
    assert {out[k]["name_core"] for k in out} == {"patterson chadwick"}
    # Ordinal noise, leading zeros and abbreviations collapse to one form.
    assert out["S1-1"]["addr_norm"] == out["S2-1"]["addr_norm"] == "1673 218 ave buckeye az"
    # Full state name maps to the same code as the abbreviation.
    assert out["S3-1"]["addr_norm"].endswith("az")
    assert out["S1-1"]["addr_nums"] == ["1673", "218"]


def test_dba_domain_and_leet_noise():
    out = norm([
        ("S3-1", "Dovasol doing business as Pediatric Dentistry Center", "1 Main St, X, UT", "US"),
        ("S3-2", "Synis Smart | www.synissmar.com", "", "US"),
        ("S3-3", "5arasaksh Consulting Pvt.", "", "India"),
        ("S2-4", "M/s DESIGN ACADEMY CORP", "", "US"),
        ("S2-5", "Trisha Thompson, P.C.", "", "US"),
    ])
    assert out["S3-1"]["name_alt"] == "pediatric dentistry center"
    assert out["S3-2"]["name_core"] == "synis smart"
    assert out["S3-3"]["name_core"] == "sarasaksh consulting"
    assert out["S2-4"]["name_core"] == "design academy"
    assert out["S2-5"]["name_core"] == "trisha thompson"


def test_indic_state_and_missing_address():
    out = norm([
        ("S1-1", "Star Media Pvt Ltd", "Room 35, Mumbai, Maharashtra", "India"),
        ("S3-1", "Star Media Pvt Ltd", "Room 35, Mumbai, महाराष्ट्र", "India"),
        ("S2-2", "Custer Offshore LLC", None, "US"),
        ("S2-3", "Custer Offshore LLC", "NULL", "US"),
    ])
    assert out["S1-1"]["addr_norm"] == out["S3-1"]["addr_norm"]
    assert out["S2-2"]["has_addr"] is False and out["S2-2"]["addr_norm"] == ""
    assert out["S2-3"]["has_addr"] is False


def test_unknown_country_is_kept_and_generic():
    out = norm([("S1-9", "Acme GmbH", "Hauptstrasse 5, Berlin", "Germany")])
    assert out["S1-9"]["ckey"] == "germany"
    assert "5" in out["S1-9"]["addr_nums"]


def test_empty_and_punctuation_only_name():
    out = norm([("S2-1", "", "", "US"), ("S2-2", "--", "##", "US")])
    assert out["S2-1"]["name_core"] == ""
    assert out["S2-2"]["name_core"] == ""
    assert out["S2-2"]["has_addr"] is False


def test_deterministic():
    rows = [("S1-1", "Tech Creative Industries Private Limited", "H1, 137 Vikaspuri, Delhi", "India")]
    assert norm(rows) == norm(rows)


def test_phonetic_key_bridges_romanized_devanagari():
    from entity_forge.normalize import phonetic_expr

    pairs = [
        ("life finance", "laiph phainens"),
        ("lotus international", "lots intrnesnl"),
        ("white indo service", "vhait indo srvisej"),
        ("maharashtra", "mharastr"),
    ]
    df = pl.DataFrame({"en": [a for a, _ in pairs], "hi": [b for _, b in pairs]}).select(
        phonetic_expr(pl.col("en")).alias("en"), phonetic_expr(pl.col("hi")).alias("hi")
    )
    assert df["en"].to_list() == df["hi"].to_list()
