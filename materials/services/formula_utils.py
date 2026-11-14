"""Shared primitives for stoichiometric-coefficient parsing and rendering.

Every formula parser in the package captures counts with the same group,
`(\\d*\\.?\\d*)`; this module is the single definition of how that capture is
converted and re-rendered, and callers import it rather than reimplementing it.

Two properties of the group drive the design:

1. `(\\d*\\.?\\d*)` matches a bare "." (from a truncated formula such as "Pb.I3",
   or from prose ending in a period), and `float(".")` raises `ValueError`.
   `coefficient()` maps such a capture to 1.

2. Re-rendering a scaled count with `%g` emits scientific notation outside
   [1e-5, 1e6], and the token regex then re-reads the mantissa as the count
   (`Pb(Sn0.00001)3` would become `PbSn3e-05`, read as Sn = 3). Fixed-point `%.6f`
   avoids the notation but truncates (0.8333333333333334 -> 0.833333) and floors
   anything below 5e-7 to zero. `format_coefficient()` renders positionally to
   twelve significant digits instead.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

__all__ = [
    "coefficient",
    "format_coefficient",
    "expand_groups",
    "COEFFICIENT_GROUP",
    "STRICT_COEFFICIENT_GROUP",
    "PSEUDO_CATIONS",
    "SPACER_RUN",
    "iter_tokens",
    "MAX_FORMULA_LEN",
    "FORMULA_CHARSET_RE",
    "FormulaExpansionLimitError",
]

#: Length and charset a string must satisfy to be treated as a chemical formula.
#:
#: Defined once for every consumer: formula candidates pulled out of free text and the
#: caller-supplied `formula_hint` of `RAGChatService.chat`. 64 is well above any real
#: formula (the longest in MP, JARVIS and PerovskiteDB has 43 characters) while keeping prose
#: out of `filter(formula__iexact=...)`.
MAX_FORMULA_LEN = 64

#: Element letters, digits, parentheses, the centred dot used in hydrates and
#: adducts, hyphen and plus. Anything else means the "formula" is prose.
FORMULA_CHARSET_RE = re.compile(r"^[A-Za-z0-9().\u00b7\-+]+$")

#: Unicode sub- and superscript digits and signs. Answers can write formulas with them
#: ("(Bi0.25Sb0.75)\u2082Te\u2083"); a user who copies one back, or a model that repeats its
#: own rendering as a tool argument, means the plain formula.
_SCRIPT_DIGITS = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u208a\u208b\u207a\u207b", "01234567890123456789++--")


def plain_digits(text: str) -> str:
    """Write Unicode sub/superscript digits and signs as ASCII ("CsSnI\u2083" -> "CsSnI3").

    For a formula or a record name only: in prose a superscript is a unit's exponent ("cm\u00b2").
    """
    return (text or "").translate(_SCRIPT_DIGITS)

#: The capture group every formula token regex uses for its count, exported so callers
#: build their patterns from one definition.
COEFFICIENT_GROUP = r"(\d*\.?\d*)"

#: The same group, but requiring at least one digit, so it cannot swallow a bare ".".
#:
#: Use it where the parse decides whether a formula is well-formed, as opposed to reading a
#: number out of one already trusted. The lenient group consumes the stray dot in "Pb." as a
#: coefficient and leaves nothing for a residue check, so `coefficient()` would read the
#: malformed formula as {Pb: 1}.
STRICT_COEFFICIENT_GROUP = r"(\d*\.?\d+)?"


def coefficient(value: object) -> float:
    """Return the stoichiometric coefficient for a `COEFFICIENT_GROUP` capture.

    An absent (None, empty or whitespace) or unparseable count means one of
    this species, the chemical convention (`CsPbI3` has one Cs). Non-finite
    values also return 1.0 so `inf`/`nan` never reach a ratio or JSON output.
    A parseable zero returns 0.0 whether it is given as a string or a number.
    """
    if value is None:
        return 1.0
    if isinstance(value, str) and not value.strip():
        return 1.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return parsed if math.isfinite(parsed) else 1.0


#: Significant digits kept when rendering a scaled coefficient.
#:
#: Expansion multiplies two binary floats, so the product carries representation noise in its
#: last digits (0.83 * 3 is 2.4899999999999998, 0.00001 * 3 is 3.0000000000000004e-05), which
#: would otherwise reach a user-visible string that is also re-parsed downstream. The noise
#: appears around the 16th significant digit, so 12 removes it while staying four orders of
#: magnitude finer than any reported stoichiometry. Significant digits, not decimal places, so
#: trace dopants below 5e-7 are not floored to zero.
_SIGNIFICANT_DIGITS = 12


def format_coefficient(value: float) -> str:
    """Render a coefficient so the token regex reads back the same number.

    The output is positional (never scientific notation, which the count group
    cannot read), keeps tiny values instead of flooring them to zero, is rounded
    to `_SIGNIFICANT_DIGITS`, and uses the shortest round-tripping decimal
    ("0.51", not "0.510000000000000006"). Integral values render bare ("3").

    Raises:
        ValueError: for a negative value, which is not a valid coefficient and
            cannot be represented in the unsigned count group. Non-finite
            values render as "1".
    """
    if not math.isfinite(value):
        return "1"
    value = float(value)
    if value < 0:
        raise ValueError(
            f"negative stoichiometric coefficient {value!r}: coefficients are "
            f"non-negative and the token grammar cannot round-trip a sign"
        )
    if value.is_integer():
        return str(int(value))
    snapped = float(f"{value:.{_SIGNIFICANT_DIGITS}g}")
    if snapped.is_integer():
        return str(int(snapped))
    try:
        text = format(Decimal(repr(snapped)), "f")
    except (InvalidOperation, ValueError):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return text.rstrip("0").rstrip(".") if "." in text else text


#: Organic pseudo-cations written as two-letter ALL-CAPS tokens.
#:
#: They must be matched ahead of the generic `[A-Z][a-z]?` element pattern or
#: "FA0.5" scales as separate F and A "elements". Real elements Ba/Pa are
#: capital+lowercase, so the alternation order cannot shadow them.
PSEUDO_CATIONS = ("FA", "MA", "EA", "BA", "PA")

_CATION_ALT = "|".join(PSEUDO_CATIONS)

#: A two-letter capital run is a pseudo-cation, not the tail of an acronym, when it belongs to
#: the unbroken chain of pseudo-cations that starts its group ("BA", "FAMA", "FA0.5MA0.5"), or
#: when what follows cannot be an acronym tail (a coefficient, another pseudo-cation, or a
#: two-character element). An unconditional alternation would read the trailing "MA" of a
#: spacer acronym such as (CHMA), (PDMA), (ThMA) or (NMA) as methylammonium; gating only on what
#: follows would split a cation alone in its group, as in (BA)2(MA)3Pb4I13. The leading-chain
#: test is a variable-length lookbehind, which `re` cannot express, so tokens are scanned
#: rather than rewritten with a single `re.sub`.
#:
#: SPACER_RUN is a run of three or more capitals ending in a pseudo-cation. These are
#: spacer-cation abbreviations (PEA for phenethylammonium, PBA, CHMA, FPEA); read element by
#: element, `(PEABr)2PbI4` would parse as phosphorus plus ethylammonium bromide. Element symbols
#: are a capital plus an optional lowercase letter, and a doped composition normally writes its
#: coefficients ("K0.05FA0.95PbI3"), which breaks the run.
#:
#: Known false negative: a coefficient-free run of one-letter elements followed by a cation is
#: ambiguous ("KMAPb2I6" could be potassium + methylammonium) and is read as an opaque "KMA"
#: and rejected. Writing "K1MAPb2I6" or "K0.5MA0.5Pb2I6" selects potassium.
#:
#: The run is matched as one opaque token and is not in `PSEUDO_CATIONS`, so it fails
#: `FormulaValidator`'s known-symbol check by name ("PEA is an unsupported spacer"). Supporting
#: a spacer means defining its radius and tool scope.
SPACER_RUN = rf"[A-Z]+?(?:{_CATION_ALT})"

_SPACER_RE = re.compile(SPACER_RUN)
_CATION_RE = re.compile(rf"(?:{_CATION_ALT})")
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
_COUNT_RE = re.compile(COEFFICIENT_GROUP)
#: The strict variant, for callers deciding whether a formula is well-formed rather than
#: reading a number out of one already trusted. It cannot swallow a bare ".", so "Pb." leaves
#: the dot for a residue check.
_STRICT_COUNT_RE = re.compile(STRICT_COEFFICIENT_GROUP)
_ACRONYM_TAIL_EXCLUDER = re.compile(rf"\d|{_CATION_ALT}|[A-Z][a-z]")

#: One innermost bracketed group and its optional multiplier.
#:
#: The multiplier group requires a digit. The lenient form would consume a bare
#: "." as if it were a count (`STRICT_COEFFICIENT_GROUP` refuses that as well),
#: and a missing multiplier already means 1.
_GROUP_RE = re.compile(r"[(\[]([^()\[\]]+)[)\]](\d+(?:\.\d+)?)?")

#: Maximum nesting depth expanded. Each pass rewrites every innermost group, so the budget
#: bounds nesting depth rather than the number of groups.
_MAX_GROUP_DEPTH = 10
# Upper bound on both the input to `expand_groups` and its expanded output. The longest formula
# in the corpus has 133 characters, so 4000 is far above anything legitimate while bounding the
# work an adversarial input can cause.
_MAX_EXPANDED_LEN = 4000


class FormulaExpansionLimitError(ValueError):
    """Raised when a formula exceeds the group-expansion bounds.

    An exception rather than a truncated result, because a partially expanded
    formula silently drops multipliers and means a different composition.
    """



def expand_groups(text: str, *, max_depth: int = _MAX_GROUP_DEPTH) -> str:
    """Flatten `(...)n` and `[...]n` groups by scaling the counts inside them.

    `Pb(I0.8Br0.2)3` -> `PbI2.4Br0.6`. Brackets expand too, because the additive
    notation common in the perovskite literature, such as
    `(CsBr)0.05[FAMAPb(I0.15Br0.85)3]0.95`, otherwise keeps its bracket and the
    downstream regexes break the formula at that character.

    Raises:
        FormulaExpansionLimitError: if the input or the expansion exceeds
            `_MAX_EXPANDED_LEN`, or groups remain after `max_depth` passes.
    """
    # Bound the input as well as the growth, so a group-free oversized string
    # is also refused.
    if len(text) > _MAX_EXPANDED_LEN:
        raise FormulaExpansionLimitError(
            f"formula is {len(text)} characters, over the "
            f"{_MAX_EXPANDED_LEN}-character limit"
        )

    result = text
    for _ in range(max_depth):
        expanded = _GROUP_RE.sub(_scale_group, result)
        if expanded == result:
            break
        if _GROUP_RE.search(expanded) and _ == max_depth - 1:
            # Depth exhausted with a group still unexpanded: a partial result
            # would drop the outer multipliers, so refuse.
            raise FormulaExpansionLimitError(
                f"formula still has unexpanded groups after {max_depth} "
                "levels of nesting"
            )
        # Growth cap: a large multiplier re-rendered positionally can grow the
        # string by an order of magnitude per pass, producing long digit runs
        # for the regexes that scan the result next.
        if len(expanded) > _MAX_EXPANDED_LEN:
            raise FormulaExpansionLimitError(
                f"group expansion reached {len(expanded)} characters, over the "
                f"{_MAX_EXPANDED_LEN}-character limit"
            )
        result = expanded
    return result


def _is_spacer_acronym(run: str) -> bool:
    """Return True when `run` is a spacer abbreviation, not a chain of pseudo-cations.

    Pseudo-cations are stripped off the front of the prefix: "FA|MA" consumes
    fully (two cations), "P|EA" leaves "P" (one opaque token, PEA). Runs shorter
    than three letters are never abbreviations.
    """
    if len(run) < 3:
        return False
    prefix = run[:-2]
    while prefix:
        if prefix[:2] in PSEUDO_CATIONS:
            prefix = prefix[2:]
        else:
            return True
    return False


def iter_tokens(text: str, *, strict: bool = False):
    """Yield (token, count_string, start, end) for each token of `text`.

    The single tokenizer used by group expansion and formula validation. The
    spacer rule cannot be written as a regex alternation, so tokens are scanned
    left to right. Characters that are neither elements nor pseudo-cations are
    yielded as single-character tokens with an empty count.

    `strict` selects `STRICT_COEFFICIENT_GROUP` for well-formedness checks; token
    boundaries are the same either way.
    """
    count_re = _STRICT_COUNT_RE if strict else _COUNT_RE
    position = 0
    leading_cation_chain = True
    length = len(text)
    while position < length:
        start = position
        spacer = _SPACER_RE.match(text, position)
        if spacer is not None and _is_spacer_acronym(spacer.group(0)):
            token, position = spacer.group(0), spacer.end()
            leading_cation_chain = False
        elif spacer is not None:
            # The run is a chain of pseudo-cations: emit each member, so the
            # last one is not split by the per-token lookahead ("PEABrFAMA").
            run = spacer.group(0)
            run_start, run_end = position, spacer.end()
            count = count_re.match(text, run_end)
            raw_count = (count.group(1) or "") if count else ""
            # A trailing count belongs to the last cation of the run
            # ("FAMA2" is FA + MA2).
            last = len(run) - 2
            for offset in range(0, len(run), 2):
                yield (
                    run[offset:offset + 2],
                    raw_count if offset == last else "",
                    run_start + offset,
                    (run_end + len(raw_count)) if offset == last
                    else run_start + offset + 2,
                )
            position = run_end + len(raw_count)
            leading_cation_chain = False
            continue
        else:
            cation = _CATION_RE.match(text, position)
            # A lowercase letter after the pair means the second capital starts
            # a two-letter element ("BAs" is boron arsenide, not butylammonium).
            # This takes precedence over the leading-chain rule.
            followed_by_lowercase = (
                cation is not None
                and cation.end() < len(text)
                and text[cation.end()].islower()
            )
            take_cation = (
                bool(cation)
                and not followed_by_lowercase
                and (
                    leading_cation_chain
                    or bool(_ACRONYM_TAIL_EXCLUDER.match(text, cation.end()))
                )
            )
            if take_cation:
                token, position = cation.group(0), cation.end()
            else:
                element = _ELEMENT_RE.match(text, position)
                if element is None:
                    # Punctuation or a stray lowercase run: surfaced as a token
                    # with no count so callers can decide what it means.
                    yield text[position], "", position, position + 1
                    position += 1
                    leading_cation_chain = False
                    continue
                token, position = element.group(0), element.end()
                leading_cation_chain = False
        count = count_re.match(text, position)
        raw_count = (count.group(1) or "") if count else ""
        position += len(raw_count)
        yield token, raw_count, start, position


def _scale_group(match: "re.Match[str]") -> str:
    inner = match.group(1)
    multiplier = coefficient(match.group(2))
    out: list[str] = []
    for token, raw_count, _start, _end in iter_tokens(inner):
        if not token.isalnum():
            out.append(token)
            continue
        scaled = coefficient(raw_count) * multiplier
        if not math.isfinite(scaled):
            out.append(token)
            continue
        # A count of exactly 1 renders bare: `Pb(I)3` -> `PbI3`.
        out.append(token if scaled == 1.0 else f"{token}{format_coefficient(scaled)}")
    return "".join(out)


#: Tokens of a plain formula read exactly: an element or an organic-cation
#: acronym, each with an optional written coefficient.
STRICT_CATIONS = ("PEA", "PPA", "FA", "MA", "EA", "GA", "BA", "PA")
STRICT_TOKEN_RE = re.compile(r"(PEA|PPA|FA|MA|EA|GA|BA|PA|[A-Z][a-z]?)(\d+(?:\.\d+)?|\.\d+)?")


def strict_composition_exact(formula: str) -> dict | None:
    """Return {token: amount} for a plain formula, or None.

    A plain formula consists only of element or organic-cation tokens with optional coefficients;
    anything else (parentheses, joined phases "|", placeholders, unknown symbols) gives None, so
    an order-equivalent match is only claimed for a plain formula.

    Amounts are the decimals as written, summed exactly with no float and no rounding, so
    FA0.1500001 and FA0.1500002 stay two amounts and a trace dopant keeps its level. `Decimal`
    equality ignores trailing zeros: 0.150 == 0.15, 1.0 == 1.

    Shared by rag's composition matching and `corpus_selection`'s member keys.
    """
    from materials.services.analysis_tools import FormulaValidator

    text = (formula or "").strip()
    if not text or not re.fullmatch(r"(?:" + STRICT_TOKEN_RE.pattern + r")+", text):
        return None
    out: dict = {}
    known = set(getattr(FormulaValidator, "_KNOWN_ELEMENTS", ()) or ())
    for token, amount in STRICT_TOKEN_RE.findall(text):
        if known and token not in known and token not in STRICT_CATIONS:
            return None
        out[token] = out.get(token, Decimal(0)) + (Decimal(amount) if amount else Decimal(1))
    return out


def decimal_text(amount: Decimal) -> str:
    """Write an exact amount in its shortest positional form: 0.1500 -> "0.15", 1E+1 -> "10"."""
    return format(amount.normalize(), "f")
