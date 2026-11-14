"""Which solar cell a material's performance is reported from.

A material can own thousands of device records, and every surface that reports
"the PCE of X" has to pick one. The reported cell is the highest-PCE cell that
passes a plausibility screen (the "champion", the usual convention for
reporting a material's device performance).
`MaterialSerializer` and the RAG tools both use this module, so they report
the same cell.

The screen and the selection are separate functions:
`is_plausible_cell` decides whether a row's own numbers are self-consistent,
`pick_champion` decides which of a candidate list is reported. Callers that hold
a pre-filtered candidate set (a PCE-window query, an in-memory prefetch cache)
use `pick_champion` directly; callers holding only a material use
`champion_cell`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from django.db.models.functions import Coalesce

__all__ = [
    "ONE_SUN_MW_CM2",
    "CellScreen",
    "cell_irradiance",
    "is_one_sun_complete_cell",
    "one_sun_cell_q",
    "pce_basis",
    "MAX_PLAUSIBLE_PCE",
    "plausible_cell_q",
    "IV_IDENTITY_TOLERANCE",
    "CHAMPION_POOL_ATTR",
    "CHAMPION_POOL_SIZE",
    "is_plausible_cell",
    "sorted_solar_cells",
    "imported_solar_cells",
    "pick_champion",
    "champion_cell",
    "champion_pool_queryset",
    "eligible_cells",
    "eligible_device_q",
    "is_device_eligible",
    "mark_eligibility",
]

#: Shockley-Queisser single-junction ceiling, with headroom for the concentrator
#: and tandem rows the corpus legitimately contains.
MAX_PLAUSIBLE_PCE = 34.0

#: The ceiling under indoor light. Indoor (LED, fluorescent) spectra are narrow, so
#: their detailed-balance limit is far above the solar one -- about 57 % at a
#: 1.9 eV gap under a white LED (Freunek et al., IEEE J. Photovolt. 3, 59, 2013) --
#: and the corpus records J-V-consistent MAPbI3 cells at 35-36 % under
#: 0.1-0.3 mW/cm². Headroom as for MAX_PLAUSIBLE_PCE.
MAX_PLAUSIBLE_PCE_BELOW_ONE_SUN = 60.0

#: Tolerance on PCE[%] = Voc[V] x Jsc[mA/cm2] x FF at 1-sun AM1.5G. Curated
#: sources round their IV columns independently, so exact agreement is not
#: expected; a factor of two apart is a transcription error, not rounding.
IV_IDENTITY_TOLERANCE = 2.0


# Device quality
#
# A `suspect` DeviceQualityIssue, or a not-single-cell one
# (`quality.NOT_SINGLE_CELL_CODES`: a module whose J-V is not per cell), takes
# the device out of every selection this module makes: the plausibility screen,
# the one-sun screen, a stated CellScreen, the champion including its fallback,
# and the SQL forms of each. The numeric screen alone is not enough: a fill
# factor back-calculated from a Jsc copied from the PCE agrees with the other
# numbers exactly and passes every identity check.

#: Set on a Device instance once its eligibility is known, so a loop over a
#: pool asks the database once rather than once per cell. `eligible_cells`
#: annotates it in SQL; `mark_eligibility` sets it in Python.
SUSPECT_ATTR = "quality_suspect"


def _suspect_issue():
    from django.db.models import OuterRef
    from materials.models import DeviceQualityIssue
    from materials.services.quality import excluding_device_issue_q
    return DeviceQualityIssue.objects.filter(excluding_device_issue_q(), device_id=OuterRef("pk"))


def eligible_device_q(prefix: str = ""):
    """The device carries no suspect objection, as a filter condition.

    `prefix` is the relation path to the Device ("devices__" from Material).
    Across a relation the test is a positive membership of the joined row, so
    it stays on the same device as the other conditions of the same
    `filter()` call; a negated condition there would read "no device of the
    material is suspect" instead.
    """
    from django.db.models import Exists, Q
    if not prefix:
        return Q(~Exists(_suspect_issue()))
    from materials.models import Device
    return Q(**{f"{prefix}id__in": Device.objects.filter(~Exists(_suspect_issue())).values("id")})


def eligible_cells(queryset):
    """`queryset` (of Device) without suspect devices, each row marked eligible."""
    from django.db.models import Exists
    return queryset.annotate(**{SUSPECT_ATTR: Exists(_suspect_issue())}).filter(
        **{SUSPECT_ATTR: False})


def mark_eligibility(devices) -> list:
    """Load the eligibility of every device in `devices` with one query."""
    devices = list(devices)
    unknown = [d for d in devices
               if getattr(d, SUSPECT_ATTR, None) is None
               and "quality_issues" not in (getattr(d, "_prefetched_objects_cache", None) or {})
               and getattr(d, "pk", None) is not None
               and hasattr(type(d), "quality_issues")]
    if unknown:
        from materials.models import DeviceQualityIssue
        from materials.services.quality import excluding_device_issue_q
        bad = set(DeviceQualityIssue.objects.filter(
            excluding_device_issue_q(), device_id__in=[d.pk for d in unknown])
            .values_list("device_id", flat=True))
        for d in unknown:
            try:
                setattr(d, SUSPECT_ATTR, d.pk in bad)
            except AttributeError:
                pass
    return devices


def is_device_eligible(device) -> bool:
    """No suspect objection on `device`; one query only when nothing is known.

    A row that is not a stored Device (an unsaved instance or another object
    with the same fields) cannot carry an objection and is eligible.
    """
    known = getattr(device, SUSPECT_ATTR, None)
    if known is not None:
        return not known
    cache = getattr(device, "_prefetched_objects_cache", None) or {}
    if "quality_issues" in cache:
        from materials.services import quality
        return quality.is_device_eligible(device)
    if getattr(device, "pk", None) is None or not hasattr(type(device), "quality_issues"):
        return True
    mark_eligibility([device])
    return not getattr(device, SUSPECT_ATTR, False)


def is_plausible_cell(device) -> bool:
    """Whether a solar cell's reported IV numbers are self-consistent.

    Applied before the champion is chosen. A row is rejected when its efficiency
    exceeds the detailed-balance limit, or when all four IV quantities are
    present and the efficiency disagrees with their product by more than
    `IV_IDENTITY_TOLERANCE`. Rows that simply lack Voc/Jsc/FF are kept - most of
    the corpus reports PCE alone, and absence is not evidence of an error.
    A device with a suspect `DeviceQualityIssue` is never plausible.
    """
    if not is_device_eligible(device):
        return False
    pce = device.pce
    if pce is None or pce <= 0:
        return False
    if pce > MAX_PLAUSIBLE_PCE:
        return False
    voc, jsc, ff = device.voc, device.jsc, device.ff
    if None in (voc, jsc, ff) or voc <= 0 or jsc <= 0 or ff <= 0:
        return True
    implied = voc * jsc * ff
    if implied <= 0:
        return True
    ratio = pce / implied
    return 1 / IV_IDENTITY_TOLERANCE <= ratio <= IV_IDENTITY_TOLERANCE


def plausible_cell_q(prefix: str = ""):
    """`is_plausible_cell` as a queryset filter, for range queries.

    The Python screen runs on loaded rows; a PCE-window query has to screen in
    SQL, or the range filter matches cells the champion screen rejects and
    returns materials whose published PCE lies outside the requested window.

    `prefix` is the relation path to the Device from the queryset's model, e.g.
    "devices__" when filtering Material.

    Mirrors `is_plausible_cell` clause for clause: positive PCE, at or below the
    detailed-balance ceiling, and, only when all four IV quantities are present
    and positive, an efficiency within IV_IDENTITY_TOLERANCE of their product.
    Rows that simply lack Voc/Jsc/FF are kept, because absence is not evidence
    of an error.
    """
    from django.db.models import F, Q

    p = prefix
    positive_pce = Q(**{f"{p}pce__gt": 0, f"{p}pce__lte": MAX_PLAUSIBLE_PCE})

    # Any of the three missing (or non-positive) means the identity cannot be
    # checked, and the row is kept.
    incomplete_iv = (
        Q(**{f"{p}voc__isnull": True})
        | Q(**{f"{p}jsc__isnull": True})
        | Q(**{f"{p}ff__isnull": True})
        | Q(**{f"{p}voc__lte": 0})
        | Q(**{f"{p}jsc__lte": 0})
        | Q(**{f"{p}ff__lte": 0})
    )
    implied = F(f"{p}voc") * F(f"{p}jsc") * F(f"{p}ff")
    consistent_iv = Q(
        **{
            f"{p}pce__gte": implied / IV_IDENTITY_TOLERANCE,
            f"{p}pce__lte": implied * IV_IDENTITY_TOLERANCE,
        }
    )
    return positive_pce & (incomplete_iv | consistent_iv) & eligible_device_q(p)


#: Efficiencies are compared at one sun: AM1.5G, 100 mW/cm², the condition
#: most corpus cells record ("AM 1.5; 100.0 mW/cm²; N2").
ONE_SUN_MW_CM2 = 100.0


#: `Device.illumination_class` as stored (one sun is 95-105 mW/cm²).
ONE_SUN_CLASS = "one-sun"
OTHER_ILLUMINATION_CLASS = "other"


def cell_irradiance(device):
    """The stored illumination of a cell in mW/cm² (`Device.light_intensity_mw_cm2`), or None.

    None means the source states no illumination. `test_condition` is display text and is not read.
    """
    return getattr(device, "light_intensity_mw_cm2", None)


def _iv_consistent(device, factor=IV_IDENTITY_TOLERANCE, irradiance=ONE_SUN_MW_CM2):
    """PCE[%] = Voc[V] x Jsc[mA/cm2] x FF / Pin[mW/cm2] x 100, within `factor`."""
    implied = device.voc * device.jsc * device.ff * 100.0 / irradiance
    return implied > 0 and 1 / factor <= device.pce / implied <= factor


def _full_jv(device):
    return all(v is not None and v > 0 for v in (device.pce, device.voc, device.jsc, device.ff))


#: Indoor light: well below any outdoor or simulator condition (0.01-1 mW/cm²
#: under LED/fluorescent lamps); 95-99 mW/cm² simulators are solar light.
INDOOR_MAX_MW_CM2 = 10.0


def _ceiling(irradiance):
    return MAX_PLAUSIBLE_PCE_BELOW_ONE_SUN if irradiance is not None and irradiance <= INDOOR_MAX_MW_CM2 \
        else MAX_PLAUSIBLE_PCE


def is_one_sun_complete_cell(device) -> bool:
    """Taken at one sun (`illumination_class`), with a positive, consistent PCE/Voc/Jsc/FF set."""
    if getattr(device, "illumination_class", None) != ONE_SUN_CLASS:
        return False
    if not _full_jv(device) or device.pce > MAX_PLAUSIBLE_PCE:
        return False
    return _iv_consistent(device) and is_device_eligible(device)


def one_sun_cell_q(prefix: str = ""):
    """`is_one_sun_complete_cell` as a queryset filter."""
    from django.db.models import F, Q

    p = prefix
    implied = F(f"{p}voc") * F(f"{p}jsc") * F(f"{p}ff")
    return (Q(**{f"{p}illumination_class": ONE_SUN_CLASS})
            & Q(**{f"{p}pce__gt": 0, f"{p}pce__lte": MAX_PLAUSIBLE_PCE,
                   f"{p}voc__gt": 0, f"{p}jsc__gt": 0, f"{p}ff__gt": 0})
            & Q(**{f"{p}pce__gte": implied / IV_IDENTITY_TOLERANCE,
                   f"{p}pce__lte": implied * IV_IDENTITY_TOLERANCE})
            & eligible_device_q(p))


def pce_basis(device):
    """What the published PCE of `device` rests on."""
    if device is None:
        return None
    if is_one_sun_complete_cell(device):
        return "one_sun"
    if not is_plausible_cell(device):
        # The unscreened fallback of a material with no plausible cell: its
        # own numbers disagree (or PCE is 0 or above the ceiling). This is the
        # predicate `pick_champion` ranks by, so a label never contradicts the
        # choice; it reads the J-V identity at one sun.
        return "implausible_record"
    illumination = getattr(device, "illumination_class", None)
    if illumination == ONE_SUN_CLASS:
        return "one_sun_incomplete_jv"
    if illumination == OTHER_ILLUMINATION_CLASS:
        return "other_illumination"
    return "illumination_not_recorded"


_NUMBER_WORDS = {"two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0, "ten": 10.0}
_IRRADIANCE_TOLERANCE = 1e-6


def _plain(value: float) -> str:
    """Format a bound without scientific notation (0.7, 1.1, 25)."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class CellScreen:
    """Conditions a question states for the cells it counts. Applied strictly.

    The platform's plausibility rule is kept for any condition the question
    does not state: without a PCE bound the ceiling is MAX_PLAUSIBLE_PCE, and
    without a factor the J-V identity holds within IV_IDENTITY_TOLERANCE, so a
    screen that states only "record a positive PCE, Voc, Jsc and FF" still
    rejects a transcription error.
    """

    irradiance_mw_cm2: Optional[float] = None
    require_full_jv: bool = False
    pce_max: Optional[float] = None
    iv_factor: Optional[float] = None
    #: Conditions stated but impossible (a factor below 1, no light): not applied, and said so.
    ignored: tuple = ()
    #: Windows on the cell's own metrics, all held by the same cell: (column, lo, lo_inclusive, hi,
    #: hi_inclusive). "FF above 0.7, top 3 by PCE" ranks each material by its best cell with FF
    #: above 0.7, never by a champion that fails the window.
    metric_windows: tuple = ()

    def _bound(self):
        # Without a stated bound, use the ceiling for the stated light.
        return self.pce_max if self.pce_max is not None else _ceiling(self.irradiance_mw_cm2)

    def _factor(self):
        # A stated factor gets a relative 1e-9 margin so a cell whose PCE equals
        # Voc x Jsc x FF up to float rounding passes in both SQL and Python.
        return self.iv_factor * (1 + 1e-9) if self.iv_factor is not None else IV_IDENTITY_TOLERANCE

    def _light(self):
        return self.irradiance_mw_cm2 or ONE_SUN_MW_CM2

    def passes(self, device) -> bool:
        if device.pce is None or device.pce <= 0 or device.pce > self._bound():
            return False
        if not is_device_eligible(device):
            return False
        recorded = cell_irradiance(device)
        if self.irradiance_mw_cm2 is not None and (
                recorded is None or abs(recorded - self.irradiance_mw_cm2) > _IRRADIANCE_TOLERANCE):
            return False
        if self.require_full_jv and not _full_jv(device):
            return False
        if _full_jv(device) and not _iv_consistent(device, self._factor(), self._light()):
            return False
        for column, lo, lo_inclusive, hi, hi_inclusive in self.metric_windows:
            value = getattr(device, column, None)
            if value is None:
                return False
            if lo is not None and (value < lo or (value == lo and not lo_inclusive)):
                return False
            if hi is not None and (value > hi or (value == hi and not hi_inclusive)):
                return False
        return True

    def apply(self, queryset, prefix: str = ""):
        """`passes` on a queryset; the stored illumination (`light_intensity_mw_cm2`) is
        compared within the same tolerance."""
        from django.db.models import F, Q

        p = prefix
        qs = queryset.filter(eligible_device_q(p), **{f"{p}pce__gt": 0, f"{p}pce__lte": self._bound()})
        full = Q(**{f"{p}voc__gt": 0}) & Q(**{f"{p}jsc__gt": 0}) & Q(**{f"{p}ff__gt": 0})
        if self.require_full_jv:
            qs = qs.filter(full)
        implied = F(f"{p}voc") * F(f"{p}jsc") * F(f"{p}ff") * (100.0 / self._light())
        consistent = Q(**{f"{p}pce__gte": implied / self._factor(), f"{p}pce__lte": implied * self._factor()})
        qs = qs.filter(consistent if self.require_full_jv else (~full | consistent))
        for column, lo, lo_inclusive, hi, hi_inclusive in self.metric_windows:
            qs = qs.filter(**{f"{p}{column}__isnull": False})
            if lo is not None:
                qs = qs.filter(**{f"{p}{column}__{'gte' if lo_inclusive else 'gt'}": lo})
            if hi is not None:
                qs = qs.filter(**{f"{p}{column}__{'lte' if hi_inclusive else 'lt'}": hi})
        if self.irradiance_mw_cm2 is not None:
            qs = qs.filter(**{
                f"{p}light_intensity_mw_cm2__gte": self.irradiance_mw_cm2 - _IRRADIANCE_TOLERANCE,
                f"{p}light_intensity_mw_cm2__lte": self.irradiance_mw_cm2 + _IRRADIANCE_TOLERANCE})
        return qs

    def describe(self) -> str:
        parts = []
        if self.irradiance_mw_cm2 is not None:
            level = f"{self.irradiance_mw_cm2:.4f}".rstrip("0").rstrip(".")
            parts.append(f"measured under {level} mW/cm²")
        if self.require_full_jv:
            parts.append("positive PCE, Voc, Jsc and FF recorded")
        parts.append(f"PCE at most {self._bound()}%")
        factor = self.iv_factor if self.iv_factor is not None else IV_IDENTITY_TOLERANCE
        parts.append(f"PCE within a factor of {factor} of Voc x Jsc x FF (scaled to the illumination)")
        for column, lo, lo_inclusive, hi, hi_inclusive in self.metric_windows:
            ends = [f"{column} {'>=' if lo_inclusive else '>'} {_plain(lo)}" if lo is not None else "",
                    f"{column} {'<=' if hi_inclusive else '<'} {_plain(hi)}" if hi is not None else ""]
            parts.append(" and ".join(e for e in ends if e) + " (on the same cell)")
        parts += [f"not applied: {note}" for note in self.ignored]
        return "; ".join(parts)

    @staticmethod
    def merge(question: Optional["CellScreen"], call: Optional["CellScreen"]):
        """Merge the question's screen with the model's call; return (screen, conflicts).

        Conditions stated in the question take precedence; the call only fills
        conditions the question leaves unstated. Each disagreement is listed.
        """
        if question is None or call is None:
            return (call or question), []
        fields, conflicts = {}, []
        for name in ("irradiance_mw_cm2", "require_full_jv", "pce_max", "iv_factor"):
            q, c = getattr(question, name), getattr(call, name)
            given_q = q not in (None, False)
            given_c = c not in (None, False)
            fields[name] = q if given_q else c
            if given_q and given_c and q != c:
                conflicts.append(f"{name}: question {q}, call {c} (question applied)")
        fields["ignored"] = tuple(dict.fromkeys(question.ignored + call.ignored))
        return CellScreen(**fields), conflicts

    @classmethod
    def from_text(cls, text: str) -> Optional["CellScreen"]:
        """Parse the cell conditions stated in `text`, or return None if there are none.

        An illumination is applied only when exactly one is stated; negated
        ones ("not one sun") are ignored. Impossible conditions (a factor below
        1, no light) are recorded in `ignored` instead of applied.
        """
        t = (text or "").replace("×", "x").replace("−", "-").replace("⁻²", "-2")
        lights = []
        for m in re.finditer(r"(?:under|at|of)\s+(\d+(?:\.\d+)?)\s*mW\s*(?:/\s*cm|\s*cm\s*-\s*2)", t, re.I):
            lights.append((m.start(), float(m.group(1))))
        # Korean phrasing: the illumination followed by a postposition or noun.
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*mW\s*/\s*cm(?:2|²|\^2)?\s*(?:조건|에서|하에서|조사|광)", t, re.I):
            if not any(abs(start - m.start()) < 12 for start, _ in lights):
                lights.append((m.start(), float(m.group(1))))
        for m in re.finditer(r"\b(one|\d+(?:\.\d+)?)[\s-]suns?\b", t, re.I):
            lights.append((m.start(), ONE_SUN_MW_CM2 * (1.0 if m.group(1).lower() == "one" else float(m.group(1)))))
        for m in re.finditer(r"\bAM\s*1\.5\s*G?\b", t, re.I):
            lights.append((m.start(), ONE_SUN_MW_CM2))
        # A light preceded by a negation or comparison ("not one sun", "below 1 sun",
        # "excluding 1-sun cells") is not a stated condition.
        negated = re.compile(r"\b(?:not|non|without|excluding|except|other\s+than|instead\s+of|rather\s+than|"
                             r"below|above|less\s+than|more\s+than|lower\s+than|higher\s+than)\b[\s(-]*"
                             r"(?:[a-z]+\s+){0,2}?(?:(?:an?|the|at|under)\s+)?$", re.I)
        stated = {v for start, v in lights if not negated.search(t[max(0, start - 40):start])}
        irradiance = stated.pop() if len(stated) == 1 else None
        full = bool(re.search(r"(?:record|report)s?\s+(?:a\s+)?positive\s+PCE,?\s+Voc,?\s+Jsc,?\s+and\s+FF", t, re.I))
        m_max = re.search(r"PCE\s+(?:is\s+|of\s+)?(?:at\s+most|no\s+higher\s+than|not\s+above|≤|<=)\s*"
                          r"(\d+(?:\.\d+)?)\s*%", t, re.I)
        m_fac = re.search(r"within\s+a\s+factor\s+of\s+(two|three|four|five|ten|\d+(?:\.\d+)?)", t, re.I)
        factor = None
        if m_fac:
            word = m_fac.group(1).lower()
            factor = _NUMBER_WORDS.get(word) or float(word)
        ignored = []
        if irradiance is not None and irradiance <= 0:
            ignored.append(f"an illumination of {irradiance:.4f}".rstrip("0").rstrip(".") + " mW/cm² (no light)")
            irradiance = None
        if factor is not None and factor < 1:
            ignored.append(f"a factor of {m_fac.group(1)} (a factor is at least 1)")
            factor = None
        if irradiance is None and not full and not m_max and factor is None and not ignored:
            return None
        return cls(irradiance_mw_cm2=irradiance, require_full_jv=full,
                   pce_max=float(m_max.group(1)) if m_max else None, iv_factor=factor, ignored=tuple(ignored))


def pick_champion(cells):
    """The reported cell, given a candidate list.

    Selecting by max(PCE) with no screen prefers a data-entry error, because a
    mis-transcribed efficiency tends to be the largest. Screen first, then take
    the maximum of what survives; fall back to the unscreened maximum when
    nothing does, so a material with only implausible records still reports
    something.

    The fallback is to the eligible cells only: a device with a suspect
    `DeviceQualityIssue` is never reported, so a material whose cells are all
    suspect reports none.
    """
    cells = [d for d in mark_eligibility(cells) if is_device_eligible(d)]
    if not cells:
        return None
    plausible = [d for d in cells if is_plausible_cell(d)]
    # One sun first: a plausible cell measured under other light (or none
    # recorded) is published only when the material has no one-sun cell with a
    # complete J-V set, so an indoor cell never outranks one-sun cells.
    one_sun = [d for d in plausible if is_one_sun_complete_cell(d)]
    return min(one_sun or plausible or cells, key=lambda d: (-(d.pce or 0), d.id))


def sorted_solar_cells(devices):
    """In-memory twin of `imported_solar_cells`, for an already-loaded cache.

    The sort key mirrors `Coalesce("pce", 0.0).desc(), "id"` exactly:
    PostgreSQL places NULLs first under a bare DESC, while the policy reads a
    missing PCE as zero.
    """
    cells = [d for d in devices if d.device_type == "Solar cell" and d.created_by_id is None]
    return sorted(
        (d for d in mark_eligibility(cells) if is_device_eligible(d)),
        key=lambda d: (-(d.pce or 0.0), d.id),
    )


def imported_solar_cells(material):
    """The candidate population, ordered so the first plausible row wins."""
    return eligible_cells(
        material.devices.filter(device_type="Solar cell", created_by__isnull=True)
    ).order_by(Coalesce("pce", 0.0).desc(), "id")


def champion_pool_queryset():
    """The Device queryset a `CHAMPION_POOL_ATTR` prefetch slices: the same
    population and order as `imported_solar_cells`, for every material."""
    from materials.models import Device
    return eligible_cells(
        Device.objects.filter(device_type="Solar cell", created_by__isnull=True)
    ).order_by(Coalesce("pce", 0.0).desc(), "id")


#: How many PCE-ordered cells the bounded lookup reads.
#:
#: The pool is ordered by descending PCE, so its first plausible row is the
#: global plausible champion and the bound is exact whenever any of the top K
#: survives the screen. Only when none does, and the pool came back full (so it
#: may have been truncated), is an unbounded read needed.
CHAMPION_POOL_SIZE = 200

#: Where the resolved champion is memoised on a Material instance.
_CACHE_ATTR = "_champion_cell_cache"
#: Champions under stated conditions, keyed by the screen.
_SCREEN_CACHE_ATTR = "_champion_cell_screen_cache"


def champion_cell(material, *, candidates=None, screen: Optional[CellScreen] = None):
    """Return the reported cell for `material`, reading a bounded pool by default.

    `candidates` restricts the population for callers that have already
    filtered it; such sets are used whole, since the filter need not preserve
    the PCE ordering the bound relies on. `screen` applies stated conditions
    strictly, with no fallback to implausible cells.
    """
    if screen is not None:
        # Stated conditions: strict, no fallback, memoized per screen.
        memo = getattr(material, _SCREEN_CACHE_ATTR, None)
        if memo is None:
            memo = {}
            try:
                setattr(material, _SCREEN_CACHE_ATTR, memo)
            except AttributeError:
                pass
        if candidates is not None:
            passing = [d for d in candidates if screen.passes(d)]
            return min(passing, key=lambda d: (-(d.pce or 0), d.id)) if passing else None
        if screen not in memo:
            # PCE-ordered: the first that passes is the champion under the screen.
            memo[screen] = screen.apply(imported_solar_cells(material)).first()
        return memo[screen]
    if candidates is not None:
        return pick_champion(candidates)

    cached = getattr(material, _CACHE_ATTR, "unset")
    if cached != "unset":
        return cached
    champion = _resolve_bounded(material)
    try:
        setattr(material, _CACHE_ATTR, champion)
    except AttributeError:
        pass
    return champion


#: The `to_attr` a caller can prefetch the champion pool into.
#:
#: The same string `serializers.CHAMPION_POOL_ATTR` holds, defined here because
#: this module owns the policy and the serializer already imports from it. A
#: `to_attr` prefetch does not populate `_prefetched_objects_cache`, so the
#: pool is looked up under this name explicitly.
CHAMPION_POOL_ATTR = "_champion_pool"


def _resolve_bounded(material):
    champion, truncated = _resolve_pool(material)
    if truncated and champion is not None and not is_one_sun_complete_cell(champion):
        # The pool was truncated, so a one-sun cell, which outranks every
        # pooled cell, may lie below it.
        deeper = imported_solar_cells(material).filter(one_sun_cell_q()).first()
        if deeper is not None:
            return deeper
    return champion


def _resolve_pool(material):
    """Return (champion of the pool, whether the pool was truncated at the bound)."""
    cache = getattr(material, "_prefetched_objects_cache", {})
    if "devices" in cache:
        # The whole relation is already loaded; resolve in memory.
        return pick_champion(sorted_solar_cells(cache["devices"])), False
    prefetched = getattr(material, CHAMPION_POOL_ATTR, None)
    if prefetched is not None:
        # The prefetch has the same filter, ordering and bound as the query
        # below, so the same rules apply.
        pool = list(prefetched)
        full = len(pool) >= CHAMPION_POOL_SIZE
        if pool and any(is_plausible_cell(d) for d in pool):
            return pick_champion(pool), full
        if not full:
            return pick_champion(pool), False
        return pick_champion(list(imported_solar_cells(material))), False
    ordered = imported_solar_cells(material)
    pool = list(ordered[:CHAMPION_POOL_SIZE])
    full = len(pool) >= CHAMPION_POOL_SIZE
    if pool and any(is_plausible_cell(d) for d in pool):
        return pick_champion(pool), full
    if not full:
        # A short pool is complete, so the fallback applies to it directly.
        return pick_champion(pool), False
    # Every pooled cell failed the screen; a plausible one may lie below the bound.
    return pick_champion(list(ordered)), False


# The radiative (detailed-balance) open-circuit voltage limit

#: CODATA 2018 exact SI values.
_Q_E = 1.602176634e-19        # C
_H = 6.62607015e-34           # J s
_C_LIGHT = 2.99792458e8       # m/s
_K_B = 1.380649e-23           # J/K


def radiative_voc_limit(eg_ev, jsc_ma_cm2, temperature_k: float = 300.0) -> Optional[float]:
    """Return the radiative (detailed-balance) Voc limit in V, or None for a bad input.

        Voc_rad = (kT/q) ln(Jsc / J0 + 1)
        J0 = q 2π/(h³c²) ∫_Eg^∞ E² exp(-E/kT) dE
           = q 2π/(h³c²) kT exp(-Eg/kT) (Eg² + 2 kT Eg + 2 (kT)²)

    with Eg in joules (eV x q) and Jsc in A/m² (mA/cm² x 10). Assumes a step
    absorptance at Eg, a flat cell emitting into a hemisphere, the Boltzmann
    form of the blackbody, temperature T, and the cell's own Jsc.
    """
    import math
    try:
        eg_ev, jsc_ma_cm2, temperature_k = float(eg_ev), float(jsc_ma_cm2), float(temperature_k)
    except (TypeError, ValueError):
        return None
    if eg_ev <= 0 or jsc_ma_cm2 <= 0 or temperature_k <= 0:
        return None
    kt = _K_B * temperature_k
    eg = eg_ev * _Q_E
    j0 = (_Q_E * 2.0 * math.pi / (_H ** 3 * _C_LIGHT ** 2)) * kt * math.exp(-eg / kt) * (
        eg * eg + 2.0 * kt * eg + 2.0 * kt * kt)
    if j0 <= 0:
        return None
    return (kt / _Q_E) * math.log(jsc_ma_cm2 * 10.0 / j0 + 1.0)
