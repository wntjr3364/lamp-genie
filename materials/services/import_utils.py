"""Formula normalization shared by the agent's tools.

``reduced_formula`` is a comparison/grouping key. It never decides that
records from different repositories or different source IDs are the same
material identity.
"""
import functools
import re

#: Sources whose records are molecules. A molecular formula is not reduced:
#: C2H4 and C4H8 are different molecules and both reduce to CH2, and one
#: molecular formula is often several isomers, so a reduced key would pool
#: records that are not the same compound. Their `reduced_formula` is blank and
#: they are matched by their own formula as stored, or by source record
#: (external_id / pk). QMOF is not here: a MOF formula is a crystal's cell
#: content, reduced like any other crystal's.
MOLECULAR_FORMULA_SOURCES = frozenset({"HCEP", "HOPV", "OCELOT", "Curated-OLED"})

#: Results of `reduce_formula`.
FORMULA_OK = "ok"
FORMULA_EMPTY = "empty"
FORMULA_UNSUPPORTED = "unsupported"

#: One token as pymatgen reads it: a capital and any lower-case run.
_PMG_TOKEN = re.compile(r"[A-Z][a-z]*")
#: pymatgen reads D and T as hydrogen isotopes and folds them into H, so
#: "DMAPbI3" would lose its D; neither is a symbol a source in this corpus
#: uses for an element, so a formula carrying one is not reduced.
_NOT_ELEMENT_TOKENS = frozenset({"D", "T"})


class FormulaReduction(tuple):
    """``(reduced, status, reason)``: what `reduce_formula` made of a formula.

    ``status`` is ``ok`` (``reduced`` is pymatgen's reduced formula), ``empty``
    (nothing to reduce) or ``unsupported`` (``reduced`` is "" on purpose, and
    ``reason`` says why). A tuple so it unpacks and compares like one.
    """

    __slots__ = ()

    def __new__(cls, reduced: str, status: str, reason: str = ""):
        return super().__new__(cls, (reduced, status, reason))

    reduced = property(lambda self: self[0])
    status = property(lambda self: self[1])
    reason = property(lambda self: self[2])

    @property
    def unsupported(self) -> bool:
        return self[1] == FORMULA_UNSUPPORTED


def _real_element_symbols() -> frozenset:
    from pymatgen.core import Element
    return frozenset(e.symbol for e in Element) - _NOT_ELEMENT_TOKENS


def reduce_formula(formula, data_source: str | None = None) -> FormulaReduction:
    """The reduced formula of `formula`, or an explicit reason it has none.

    Unsupported, so ``reduced`` is "":

    * a molecular source (`MOLECULAR_FORMULA_SOURCES`);
    * a formula with a token that is not an element -- an organic cation
      abbreviation (MA, FA, GU, BA, PEA ...), a polymer name (PANI), a word
      (FALSE). pymatgen reads such tokens as dummy species and reduces the
      rest, which would put every FAxMA(1-x)PbI3 ratio under one key;
    * a formula pymatgen cannot parse.

    A record without a reduced formula is matched by its own formula (exact
    composition, any order) or by its source record, never by a reduced key.
    """
    text = "" if formula is None else str(formula).strip()
    if not text:
        return FormulaReduction("", FORMULA_EMPTY)
    if data_source in MOLECULAR_FORMULA_SOURCES:
        return FormulaReduction("", FORMULA_UNSUPPORTED, "molecular source")
    try:
        from pymatgen.core import Composition
        from pymatgen.core.periodic_table import DummySpecies
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for formula normalization / cross-source "
            "dedup. Install it with `pip install pymatgen`."
        ) from exc
    real = _real_element_symbols()
    foreign = sorted({t for t in _PMG_TOKEN.findall(text) if t not in real})
    if foreign:
        return FormulaReduction("", FORMULA_UNSUPPORTED,
                                "not an element: " + ", ".join(foreign[:5]))
    try:
        comp = Composition(text)
    except Exception:
        return FormulaReduction("", FORMULA_UNSUPPORTED, "unparseable")
    if not comp or any(isinstance(sp, DummySpecies) for sp in comp):
        return FormulaReduction("", FORMULA_UNSUPPORTED, "not an element")
    # pymatgen names eleven ratios as molecules (O -> O2, NaO -> Na2O2); a reduced formula is
    # the element ratio
    reduced = comp.reduced_formula
    return FormulaReduction(_molecule_to_ratio().get(reduced, reduced), FORMULA_OK)


@functools.lru_cache(maxsize=1)
def _molecule_to_ratio() -> dict:
    from pymatgen.core import Composition
    return {molecule: ratio for ratio, molecule in Composition.special_formulas.items()}


def normalize_formula(formula: str, data_source: str | None = None) -> str:
    """
    Normalize a chemical formula to pymatgen reduced_formula.

    Behavior:
      - Empty/None: returns "".
      - pymatgen unavailable: raises ImportError. Falling back to the raw
        input would store strings that are not reduced formulas, and formulas
        equivalent under pymatgen normalization would stop matching.
      - Unsupported (see `reduce_formula`: a token that is not an element, an
        unparseable string, a molecular source): returns "", so one
        unparseable formula raises nothing and no raw string is used as a
        reduced formula.
    """
    return reduce_formula(formula, data_source).reduced
