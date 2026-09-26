"""
Text normalisation for business names and addresses.

Design goals
------------
* Country-agnostic: nothing here branches on the `country` label, so France (unseen in
  training) goes through exactly the same code path as US / India.
* Script-agnostic: every Indic script is transliterated to Latin with one table, using the
  fact that the Unicode Indic blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
  Telugu, Kannada, Malayalam) share the same 128-code-point layout (ISCII heritage).
* Produces several "views" of each string; later stages compare views, not raw text.
"""
import re
import unicodedata

# --------------------------------------------------------------------------------------
# 1. Indic -> Latin transliteration (single table, offset-based)
# --------------------------------------------------------------------------------------
_INDIC_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_CONS = {  # offset -> consonant (without inherent vowel)
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'n', 0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j',
    0x1D: 'jh', 0x1E: 'n', 0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n', 0x24: 't',
    0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n', 0x29: 'n', 0x2A: 'p', 0x2B: 'ph', 0x2C: 'b',
    0x2D: 'bh', 0x2E: 'm', 0x2F: 'y', 0x30: 'r', 0x31: 'r', 0x32: 'l', 0x33: 'l', 0x34: 'l',
    0x35: 'v', 0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    0x58: 'q', 0x59: 'kh', 0x5A: 'g', 0x5B: 'z', 0x5C: 'd', 0x5D: 'rh', 0x5E: 'f', 0x5F: 'y',
}
_VOWEL = {  # independent vowels
    0x05: 'a', 0x06: 'aa', 0x07: 'i', 0x08: 'ii', 0x09: 'u', 0x0A: 'uu', 0x0B: 'ri', 0x0C: 'li',
    0x0D: 'e', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x11: 'o', 0x12: 'o', 0x13: 'o', 0x14: 'au',
    0x60: 'ri', 0x61: 'li',
}
_MATRA = {  # dependent vowel signs
    0x3E: 'aa', 0x3F: 'i', 0x40: 'ii', 0x41: 'u', 0x42: 'uu', 0x43: 'ri', 0x44: 'ri', 0x45: 'e',
    0x46: 'e', 0x47: 'e', 0x48: 'ai', 0x49: 'o', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au', 0x62: 'li',
    0x63: 'li',
}
_SIGN = {0x01: 'n', 0x02: 'n', 0x03: 'h', 0x50: 'om', 0x70: 'n'}
# Malayalam chillu letters (consonant with no inherent vowel) and au length mark
_CHILLU = {0x7A: 'n', 0x7B: 'n', 0x7C: 'r', 0x7D: 'l', 0x7E: 'l', 0x7F: 'k', 0x54: 'm', 0x55: 'y', 0x56: 'l'}
_SKIP = {0x57, 0x4E, 0x00, 0x3D, 0x64, 0x65}
_VIRAMA, _NUKTA = 0x4D, 0x3C


def _indic_offset(ch):
    cp = ord(ch)
    for base in _INDIC_BLOCKS:
        if base <= cp < base + 0x80:
            return cp - base
    return None


def transliterate(text: str) -> str:
    """Transliterate any Indic-script characters to Latin; leave everything else alone."""
    if not text or all(ord(c) < 0x0900 for c in text):
        return text
    # Malayalam conjuncts whose reading differs from the component letters
    text = text.replace('\u0D31\u0D4D\u0D31', '\u0D1F\u0D4D\u0D1F').replace('\u0D28\u0D4D\u0D31', '\u0D28\u0D4D\u0D1F')
    out, pending = [], False  # pending = a consonant is waiting for its inherent 'a'
    for ch in text:
        off = _indic_offset(ch)
        if off is None:
            if pending:
                out.append('a'); pending = False
            out.append(ch)
            continue
        if off in _CONS:
            if pending:
                out.append('a')
            out.append(_CONS[off]); pending = True
        elif off in _MATRA:
            out.append(_MATRA[off]); pending = False
        elif off == _VIRAMA:
            pending = False
        elif off == _NUKTA:
            continue
        elif off in _VOWEL:
            if pending:
                out.append('a'); pending = False
            out.append(_VOWEL[off])
        elif off in _SIGN:
            if pending:
                out.append('a'); pending = False
            out.append(_SIGN[off])
        elif off in _SKIP:
            continue
        elif ch >= '\u0D00' and ch < '\u0D80' and off in _CHILLU:
            if pending:
                out.append('a')
            out.append(_CHILLU[off]); pending = False
        elif 0x66 <= off <= 0x6F:  # native digits
            if pending:
                out.append('a'); pending = False
            out.append(str(off - 0x66))
        else:
            if pending:
                out.append('a'); pending = False
    if pending:
        out.append('a')
    s = ''.join(out)
    # schwa deletion at word end ("kanstrakshana" -> "kanstrakshan")
    return re.sub(r'(?<=[bcdfghjklmnpqrstvwxyz])a\b', '', s)


def has_indic(text: str) -> bool:
    return any(_indic_offset(c) is not None for c in text or '')


# --------------------------------------------------------------------------------------
# 2. Generic cleaning
# --------------------------------------------------------------------------------------


def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def base_clean(s: str) -> str:
    s = transliterate(s or '')
    s = strip_accents(s).lower()
    s = s.replace('&', ' and ').replace('+', ' and ')
    s = re.sub(r"[’'`]", '', s)          # o'brien -> obrien
    s = re.sub(r'[^a-z0-9/\-\s.,#]', ' ', s)
    return s


# --------------------------------------------------------------------------------------
# 3. Business names
# --------------------------------------------------------------------------------------
NAME_ABBR = {
    'corp': 'corporation', 'co': 'company', 'inc': 'incorporated', 'ltd': 'limited',
    'pvt': 'private', 'pvtltd': 'private limited', 'intl': 'international', 'mfg': 'manufacturing',
    'svcs': 'services', 'svc': 'service', 'assn': 'association', 'assoc': 'associates',
    'bros': 'brothers', 'dept': 'department', 'univ': 'university', 'natl': 'national',
    'mgmt': 'management', 'tech': 'technologies', 'grp': 'group', 'ent': 'enterprises',
    'cie': 'compagnie', 'ste': 'societe', 'sté': 'societe', 'fr': 'freres',
}
# legal forms / honorifics / generic fillers -> removed from the "core" name
LEGAL = {
    'incorporated', 'corporation', 'company', 'limited', 'private', 'llc', 'llp', 'pllc', 'plc',
    'lp', 'ltd', 'inc', 'pvt', 'corp', 'co', 'the', 'and', 'of', 'dba', 'mr', 'mrs', 'ms', 'dr',
    'sri', 'shri', 'shree', 'm/s', 'ms.', 'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci', 'snc',
    'scop', 'selarl', 'gmbh', 'ei', 'cie', 'compagnie', 'de', 'du', 'des', 'la', 'le', 'les', 'et', 'l', 'd', 'pa', 'pc',
}
_NAME_NOISE = [
    (re.compile(r'\((?:id|ref|no)[:\s#]*[\w-]+\)', re.I), ' '),  # "(ID: 52838)"
    (re.compile(r'#\s*\d+'), ' '),                               # "#30304"
    (re.compile(r'\|.*$'), ' '),                                 # "| www.site.com"
    (re.compile(r'\b(?:www\.)?[a-z0-9-]+\.(?:com|in|net|org|co|fr|io|biz)\b', re.I), ' '),
    (re.compile(r'\b(l)\.(l)\.(c)\.?', re.I), 'llc'),
    (re.compile(r'\b(l)\.(l)\.(p)\.?', re.I), 'llp'),
    (re.compile(r'\b(s)\.(a)\.(s)\.?', re.I), 'sas'),
    (re.compile(r'\b(e)\.(u)\.(r)\.(l)\.?', re.I), 'eurl'),
    (re.compile(r'\b(s)\.(a)\.(r)\.(l)\.?', re.I), 'sarl'),
]
_DOMAIN = re.compile(r'(?:www\.)?([a-z0-9-]+)\.(?:com|in|net|org|co|fr|io|biz)\b', re.I)
_HANDLE = re.compile(r'^[#@]([a-z0-9_]+)$', re.I)


# canonical legal-form classes (sibling distractors often differ ONLY here: "Pvt Ltd" vs "LLP")
LEGAL_CLASS = {
    'llc': 'llc', 'incorporated': 'inc', 'inc': 'inc', 'corporation': 'corp', 'corp': 'corp',
    'limited': 'ltd', 'ltd': 'ltd', 'private': 'pvt', 'pvt': 'pvt', 'llp': 'llp', 'pllc': 'pllc',
    'plc': 'plc', 'lp': 'lp', 'pc': 'pc', 'pa': 'pa', 'company': 'co', 'co': 'co', 'public': 'public',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'eurl': 'eurl', 'sa': 'sa', 'sci': 'sci', 'snc': 'snc',
    'gmbh': 'gmbh', 'ei': 'ei', 'prvt': 'pvt', 'lmtd': 'ltd', 'krprshn': 'corp', 'krprsn': 'corp', 'kmpn': 'co',
}


def legal_forms(name_full: str) -> str:
    out = set()
    for t in name_full.split():
        c = LEGAL_CLASS.get(t) or LEGAL_CLASS.get(phonetic(t))
        if c:
            out.add(c)
    return ' '.join(sorted(out))


LEGAL_PH = {'prvt', 'lmtd', 'lmt', 'nkrprtd', 'krprshn', 'krprsn', 'kmpn', 'lmtdd'}


def normalize_name(raw: str):
    """Return dict of name views."""
    raw = raw or ''
    low = strip_accents(transliterate(raw)).lower().strip()
    dom = _DOMAIN.search(low)
    hnd = _HANDLE.match(low)
    domain = (dom.group(1) if dom else (hnd.group(1) if hnd else '')).replace('-', '')
    s = low
    for pat, rep in _NAME_NOISE:
        s = pat.sub(rep, s)
    s = base_clean(s)
    s = re.sub(r'[.,/\-#]', ' ', s)
    raw_toks = s.split()
    # glue runs of single letters: "l l c" -> "llc", "s a" -> "sa", "m p" -> "mp"
    glued, buf = [], ''
    for t in raw_toks:
        if len(t) == 1 and t.isalpha():
            buf += t
        else:
            if buf: glued.append(buf); buf = ''
            glued.append(t)
    if buf: glued.append(buf)
    toks = [NAME_ABBR.get(t, t) for t in glued]
    toks = ' '.join(toks).split()
    full = ' '.join(toks)
    core_toks = [t for t in toks if t not in LEGAL and not t.isdigit() and phonetic(t) not in LEGAL_PH]
    # de-duplicate consecutive repeats ("vidyalaya vidyalaya")
    core_toks = [t for i, t in enumerate(core_toks) if i == 0 or t != core_toks[i - 1]]
    core = ' '.join(core_toks)
    return {
        'name_full': full,
        'name_core': core,
        'name_sorted': ' '.join(sorted(core_toks)),     # order-invariant view
        'name_phon': ' '.join(phonetic(t) for t in core_toks),
        'name_concat': ''.join(core_toks),              # for domain-style names
        'domain': domain,
        'is_domain': int(bool(domain) and len(core_toks) == 0),
        'name_native': int(has_indic(raw)),
        'legal': legal_forms(full),
    }


# --------------------------------------------------------------------------------------
# 4. Phonetic skeleton (script-independent) -- makes "Construction" == "कंस्ट्रक्शन"
# --------------------------------------------------------------------------------------
_PH_RULES = [
    (r'tion', 'shn'), (r'sion', 'shn'), (r'ph', 'f'), (r'ck', 'k'), (r'q', 'k'), (r'x', 'ks'),
    (r'c(?=[eiy])', 's'), (r'c', 'k'), (r'w', 'v'), (r'z', 'j'),
    (r'(?<=[kgcjtdpb])h', ''), (r'sh', 's'), (r'ee', 'i'), (r'oo', 'u'),
]
_PH_RULES = [(re.compile(a), b) for a, b in _PH_RULES]


def phonetic(tok: str) -> str:
    t = tok
    for pat, rep in _PH_RULES:
        t = pat.sub(rep, t)
    head, tail = t[:1], re.sub(r'[aeiouy]', '', t[1:])
    t = (head if head not in 'aeiouy' else '') + tail
    return re.sub(r'(.)\1+', r'\1', t) or tok[:1]


# --------------------------------------------------------------------------------------
# 5. Addresses
# --------------------------------------------------------------------------------------
ADDR_ABBR = {
    'rd': 'road', 'st': 'street', 'str': 'street', 'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'bd': 'boulevard', 'dr': 'drive', 'ln': 'lane', 'ct': 'court',
    'cir': 'circle', 'trl': 'trail', 'hwy': 'highway', 'pkwy': 'parkway', 'pl': 'place',
    'sq': 'square', 'ter': 'terrace', 'cres': 'crescent', 'apt': 'unit', 'ste': 'unit',
    'suite': 'unit', 'apartment': 'unit', 'fl': 'floor', 'flr': 'floor', 'bldg': 'building',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west', 'ne': 'northeast', 'nw': 'northwest',
    'se': 'southeast', 'sw': 'southwest', 'mt': 'mount', 'ft': 'fort',
    'r': 'rue', 'ch': 'chemin', 'imp': 'impasse', 'all': 'allee', 'pte': 'porte',
    'opp': 'opposite', 'nr': 'near', 'nagr': 'nagar', 'mg': 'mahatma gandhi',
}
ADDR_STOP = {'no', 'door', 'h', 'hno', 'house', 'plot', 'unit', 'po', 'box', 'null', 'na',
             'c/o', 'co', 'near', 'opposite', 'and', 'the', 'of', 'de', 'du', 'des', 'la',
             'le', 'kh', 'sno', 'gala', 'shop', 'flat', 'floor', 'building', 'region'}
_NUM = re.compile(r'\d+[a-z]?(?:/\d+[a-z]?)*')


def normalize_address(raw: str):
    raw = raw or ''
    s = base_clean(raw)
    parts = [p.strip() for p in s.split(',') if p.strip()]
    s2 = re.sub(r'[.#/\-]', ' ', s.replace(',', ' , '))
    toks = [ADDR_ABBR.get(t, t) for t in s2.split() if t != ',']
    toks = ' '.join(toks).split()
    content = [t for t in toks if t not in ADDR_STOP]
    nums = _NUM.findall(s)
    nums = [n for n in nums if len(n) <= 8]
    words = [t for t in content if not t[0].isdigit()]
    return {
        'addr_full': ' '.join(toks),
        'addr_words': ' '.join(words),
        'addr_nums': ' '.join(nums),
        'house_no': nums[0] if nums else '',
        'postcode': next((n for n in nums if len(n) in (5, 6) and n.isdigit()), ''),
        'addr_parts': parts,
        'addr_empty': int(len(raw.strip()) == 0),
        'addr_native': int(has_indic(raw)),
    }


def street_and_locality(addr_parts):
    """Split a normalised address into the street part (the comma-part holding the first
    number, else the first part) and the locality words (all other parts)."""
    if not addr_parts:
        return '', ''
    si = next((i for i, p in enumerate(addr_parts) if any(ch.isdigit() for ch in p)), 0)
    def words(p):
        ts = re.sub(r'[.#/\-]', ' ', p).split()
        ts = [ADDR_ABBR.get(t, t) for t in ts]
        return [t for t in ' '.join(ts).split() if t.isalpha() and len(t) > 1 and t not in ADDR_STOP]
    street = words(addr_parts[si])
    loc = [w for i, p in enumerate(addr_parts) if i != si for w in words(p)]
    return ' '.join(street), ' '.join(loc)


if __name__ == '__main__':
    for n in ['ईस्टर्न कंस्ट्रक्शन प्राइवेट लिमिटेड', 'Eastern Construction Private Limited',
              'গ্রেট অ্যাগ্রো লিমিটেড', 'Great Agro Limited', 'Clean  Highland Forefront LLC (ID: 52838)',
              'ablecarleykeo.com', '#shieldscordova', 'WHITTLE, OLIVAS and MITCHELL L.L.C.',
              'Europ & Frères Distribution S.A.', 'SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com']:
        print(n, '->', normalize_name(n))
    for a in ['NO ##701 RATNESHWAR BLDG268 BHAGWANDAS INDRAJIT RD BANGANGA, MUMBAI, महाराष्ट्र',
              '309-313 RATTVIK CIR, BUFFALO, MN', '63 R. DE DIEPPE, LILLE, Hauts-de-France']:
        print(a, '->', normalize_address(a))
