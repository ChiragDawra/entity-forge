"""Hand-written normalization dictionaries.

Everything in this file was written by hand from general knowledge of US, Indian
and French naming/address conventions plus inspection of the *provided* training
files. No external database, gazetteer, geocoder or downloaded list is used
(competition fair-play rule). Keys are lowercase ASCII after transliteration,
except ``INDIC_STATE_NAMES`` which is matched before transliteration.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Business-name tokens
# ---------------------------------------------------------------------------

# Legal-form tokens. Removed from ``name_core`` (sources reorder, drop and
# abbreviate them freely) but kept in ``name_norm``.
LEGAL_TOKENS: frozenset[str] = frozenset(
    {
        # US / generic
        "inc", "incorporated", "corp", "corporation", "co", "company", "cos", "llc",
        "lc", "llp", "lp", "ltd", "limited", "pllc", "pc", "plc", "ltda", "lllp",
        # India
        "pvt", "private", "pvtltd", "opc",
        # India, as transliterated from Devanagari by anyascii
        "praivet", "limitid", "elelpi", "kampni", "kampani",
        # France
        "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "sca", "scop", "scp", "selarl",
    }
)

# Prefix / filler tokens that carry no identity signal.
FILLER_TOKENS: frozenset[str] = frozenset(
    {"the", "m", "s", "ms", "messrs", "sri", "shri", "shree", "sree", "and", "of",
     "le", "la", "les", "l", "de", "du", "des", "d", "et", "www", "com", "net", "org"}
)

# Markers that introduce an alternative / trade name ("X doing business as Y").
# The text after the last marker is kept as ``name_alt``.
DBA_MARKERS: tuple[str, ...] = (
    "doing business as", "formerly known as", "also known as", "trading as",
    "formerly", "d b a", "dba", "f k a", "fka", "a k a", "aka",
)

# Synonyms that should compare equal (applied to both name_norm and name_core).
NAME_TOKEN_MAP: dict[str, str] = {
    "intl": "international", "mfg": "manufacturing", "svcs": "service",
    "svc": "service", "services": "service", "mgmt": "management",
    "assoc": "associate", "assocs": "associate", "associates": "associate",
    "bros": "brothers", "ctr": "center", "centre": "center", "grp": "group",
    "hosp": "hospital", "univ": "university", "dept": "department", "natl": "national",
    "technologies": "tech", "technology": "tech", "industries": "industry",
    "inds": "industry", "enterprises": "enterprise", "traders": "trader",
    "trading": "trader", "exports": "export", "imports": "import",
    "solutions": "solution", "systems": "system", "products": "product",
    "partners": "partner", "holdings": "holding", "ventures": "venture",
    "consultants": "consultant", "consultancy": "consulting",
}

# ---------------------------------------------------------------------------
# Address tokens: long form and abbreviation map to one short canonical form
# ---------------------------------------------------------------------------

ADDRESS_TOKEN_MAP_GENERIC: dict[str, str] = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "drive": "dr", "lane": "ln", "court": "ct",
    "circle": "cir", "circ": "cir", "place": "pl", "parkway": "pkwy", "pky": "pkwy",
    "highway": "hwy", "trail": "trl", "terrace": "ter", "terr": "ter", "square": "sq",
    "crescent": "cres", "apartment": "apt", "apts": "apt", "apartments": "apt",
    "suite": "ste", "floor": "fl", "flr": "fl", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw", "mount": "mt",
    "fort": "ft", "point": "pt", "junction": "jct", "expressway": "expy",
    "freeway": "fwy", "route": "rte",
    # India
    "near": "nr", "opposite": "opp", "village": "vlg", "vill": "vlg",
    "district": "dist", "distt": "dist", "taluka": "tq", "taluk": "tq",
    "tehsil": "tq", "sector": "sec", "colony": "col", "nagar": "ngr", "marg": "mrg",
    "cross": "crs", "main": "mn", "phase": "ph", "block": "blk", "house": "h",
    "hno": "h", "khasra": "kh", "ground": "gr", "first": "1", "second": "2",
    "third": "3", "fourth": "4", "fifth": "5",
    # France
    "rue": "r", "allee": "all", "impasse": "imp", "chemin": "ch", "faubourg": "fbg",
    "quai": "qu", "residence": "res",
}

# "st" means "saint" in French addresses.
ADDRESS_TOKEN_MAP_FRANCE: dict[str, str] = {
    **ADDRESS_TOKEN_MAP_GENERIC,
    "st": "saint", "ste": "sainte", "street": "st",
}

# Placeholders and city-type suffixes that sources add/remove at random.
ADDRESS_DROP_TOKENS: frozenset[str] = frozenset(
    {"null", "none", "nan", "na", "cdp", "township", "townshp", "twp", "town", "of",
     "the", "city", "county", "po", "box", "pmb", "no", "and", "de", "du", "des",
     "la", "le", "les", "d", "l"}
)

# ---------------------------------------------------------------------------
# Regions / states -> canonical short code
# ---------------------------------------------------------------------------

US_STATES: dict[str, str] = {
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
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}

INDIA_STATES: dict[str, str] = {
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka",
    "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mnp",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "tamilnadu": "tn", "telangana": "tg", "ts": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "chd", "puducherry": "py", "pondicherry": "py",
}

# Native-script state names seen in the provided S2/S3 files (hand-written).
# Matched on the raw string, before transliteration.
INDIC_STATE_NAMES: dict[str, str] = {
    "महाराष्ट्र": "mh", "नई दिल्ली": "dl", "दिल्ली": "dl", "उत्तर प्रदेश": "up",
    "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn", "ગુજરાત": "gj", "পশ্চিমবঙ্গ": "wb",
    "తెలంగాణ": "tg", "हरियाणा": "hr", "राजस्थान": "rj", "കേരളം": "kl", "बिहार": "br",
    "मध्य प्रदेश": "mp", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb", "ଓଡ଼ିଶା": "od",
    "उत्तराखंड": "uk", "झारखंड": "jh", "छत्तीसगढ़": "cg", "गोवा": "ga",
    "हिमाचल प्रदेश": "hp", "অসম": "as", "असम": "as",
}

FRANCE_REGIONS: dict[str, str] = {
    "auvergne rhone alpes": "ara", "bourgogne franche comte": "bfc", "bretagne": "bre",
    "centre val de loire": "cvl", "corse": "cor", "grand est": "ges",
    "hauts de france": "hdf", "ile de france": "idf", "normandie": "nor",
    "nouvelle aquitaine": "naq", "occitanie": "occ", "pays de la loire": "pdl",
    "provence alpes cote d azur": "pac",
}

# Region table per lowercase country label. Unknown countries get generic
# normalization only: country is an open set.
REGION_TABLES: dict[str, dict[str, str]] = {
    "us": US_STATES,
    "india": INDIA_STATES,
    "france": FRANCE_REGIONS,
}

ADDRESS_TOKEN_MAPS: dict[str, dict[str, str]] = {
    "france": ADDRESS_TOKEN_MAP_FRANCE,
}
