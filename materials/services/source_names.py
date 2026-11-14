"""Resolve collection names without confusing a collection with its prefix."""
import re

SOURCE_ALIASES = {
    "MP": ("Materials Project", "Material Project", "MP"),
    "OQMD": ("OQMD",),
    "JARVIS": ("JARVIS",),
    "JARVIS-SC": ("JARVIS-SC", "JARVIS SC", "JARVIS superconductors"),
    "C2DB": ("C2DB",),
    "2DMatPedia": ("2DMatPedia",),
    "StarryData2": ("StarryData2", "StarryData"),
    "PerovskiteDB": ("PerovskiteDB", "perovskite database"),
    "HOPV": ("HOPV",),
    "HCEP": ("HCEP",),
    "OCELOT": ("OCELOT",),
    "QMOF": ("QMOF",),
    "MXene": ("MXene",),
    "Curated-OLED": ("Curated-OLED", "Curated OLED"),
    "Curated-Perovskite": ("Curated-Perovskite", "Curated Perovskite"),
}
_CANONICAL = {alias.casefold(): source for source, aliases in SOURCE_ALIASES.items()
              for alias in aliases}
_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(a) for a in
    sorted(_CANONICAL, key=lambda a: (-len(a), a))) + r")(?!\w)", re.I)


def without_source_names(text):
    """Blank out source names so "OLED"/"Perovskite" in a name is not read as a filter."""
    return _PATTERN.sub(" ", text or "")


class UnknownCollection(ValueError):
    """A collection name no LAMP source answers to."""


#: Further names the chat reads (the dataset's own name, HOPV15). Kept out of SOURCE_ALIASES,
#: which the answer-claim reader also uses.
_CHAT_ALIASES = {"HOPV": ("HOPV15",), "JARVIS": ("JARVIS-DFT", "JARVIS DFT")}
_CHAT_CANONICAL = {**_CANONICAL, **{alias.casefold(): source for source, aliases in _CHAT_ALIASES.items()
                                    for alias in aliases}}
#: A name followed by a Korean particle is still the name: the boundary is a non-ASCII-word
#: character, not `\w`, which counts Hangul as a word character.
#: "MP" only as written in capitals: a lowercase "mp" is a melting point.
_CHAT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:" + "|".join("(?-i:MP)" if a == "mp" else re.escape(a) for a in
    sorted(_CHAT_CANONICAL, key=lambda a: (-len(a), a))) + r")(?![A-Za-z0-9_])", re.I)


def source_spans(text):
    """Return (start, end, source) for every source name in a chat text, in order."""
    return [(m.start(), m.end(), _CHAT_CANONICAL[m.group().casefold()]) for m in _CHAT_PATTERN.finditer(text or "")]


#: The source-name pattern, for readers that match a source inside another pattern ("per MP").
CHAT_SOURCE_PATTERN = _CHAT_PATTERN.pattern


def _one_edit_apart(a, b):
    """Return True if two words differ by one substitution, insertion, deletion or transposition."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def sources_named_in(text, fuzzy=False):
    """Return the sources a chat text names, in order of first mention.

    With `fuzzy`, a word of four or more characters one edit away from a source
    name also counts ("C2BD" for C2DB). Fuzzy matching is meant for reading
    questions, not for checking answers.
    """
    found = [(m.start(), _CHAT_CANONICAL[m.group().casefold()]) for m in _CHAT_PATTERN.finditer(text or "")]
    if fuzzy:
        names = [n for n in _CHAT_CANONICAL if len(n) >= 4 and " " not in n]
        rest = _CHAT_PATTERN.sub(lambda m: " " * len(m.group()), text or "")  # keeps positions
        for m in re.finditer(r"(?<![A-Za-z0-9_])[A-Za-z0-9][A-Za-z0-9-]{3,}(?![A-Za-z0-9_])", rest):
            # Only words shaped like a source name (at least two capitals or digits, e.g. "C2BD"),
            # so ordinary words such as "hope" or "Hopf" are skipped.
            if len(re.findall(r"[A-Z0-9]", m.group())) < 2:
                continue
            # ... and only where sources are being named: another source is named exactly,
            # or a word such as "database" is nearby ("2DMatPedia vs C2BD vs JARVIS").
            near = (text or "")[max(0, m.start() - 40):m.end() + 40]
            if not found and not re.search(r"\b(?:sources?|databases?|db|datasets?|collections?)\b|출처|데이터베이스|DB", near, re.I):
                continue
            hits = {_CHAT_CANONICAL[n] for n in names if _one_edit_apart(m.group().casefold(), n)}
            if len(hits) == 1:
                found.append((m.start(), hits.pop()))
    return list(dict.fromkeys(source for _, source in sorted(found)))


#: Words around a source's name that do not change which source it is ("the Materials Project
#: database", "the MP dataset").
_REMARK_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")  # "(Jacobsson)", "(2021)", "(mp-149)"
_FILLER_RE = re.compile(
    r"\b(?:the|both|database|db|dataset|data\s*set|data|collection|corpus|repository|records?|entries|set|from|in|of|"
    r"project|version|release|v?\d+(?:\.\d+)*|2d|3d|dft|materials?|compounds?|structures?|crystals?|molecules?|"
    r"monolayers?|thermoelectrics?|chromophores?|perovskites?|mofs?|emitters?|devices?|cells?)\b|['’]s\b|[\s.()\[\]-]+",
    re.I)


def canonical_sources(names):
    """Map each name to its LAMP source(s), in order, without duplicates.

    A name built around one source name ("the Materials Project database") is
    that source; a list ("MP/OQMD", "MP and PerovskiteDB") is each of them.
    Anything else ("everything except OQMD", "non-MP", "MP-like", "MP, AFLOW",
    a blank name) raises UnknownCollection rather than guessing.
    """
    result = []
    for name in names or []:
        raw = (name or "").strip()
        source = _CHAT_CANONICAL.get(raw.casefold())
        if source:
            found = [source]
        else:
            parts = [p for p in re.split(r"\s*(?:/|,|&|;|\||\+|\b(?:and|or|vs\.?|versus)\b)\s*", _REMARK_RE.sub(" ", raw),
                                         flags=re.I) if p.strip()]
            found = []
            for part in parts:
                whole = _CHAT_CANONICAL.get(part.strip().casefold())
                named = [whole] if whole else sources_named_in(part)
                rest = "" if whole else _FILLER_RE.sub("", _CHAT_PATTERN.sub("", part))
                if len(named) != 1 or rest:
                    raise UnknownCollection(f"Unknown collection: {name}")
                found.append(named[0])
            if not found:
                raise UnknownCollection(f"Unknown collection: {name!r}")
        for source in found:
            if source not in result:
                result.append(source)
    return result
