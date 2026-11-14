"""Which stored values and devices an answer, a ranking or a statistic may use.

The corpus keeps every value a source published. A ``suspect`` objection
(`PropertyQualityIssue`, `DeviceQualityIssue`) marks a value that is shown or
strongly suspected to be wrong and takes the row out of answers, windows,
rankings, counts and statistics; the row stays stored and listed with its
objections on the record. A ``warning`` keeps the row in and is reported
beside it.

The exclusion happens before a row is selected. Filtering a window,
picking a champion or slicing the first eight properties and only then
dropping suspect rows lets a material be chosen on a suspect value and then
shown with a different one. So every path uses the same two predicates here,
as a queryset filter for SQL paths and as an object test for prefetched rows.

PerovskiteDB band gaps carry a second, independent qualifier: the source's
own basis column, stored as `Property.context.method_basis`. "Composition"
means the gap was assigned from the composition rather than measured on the
sample, so those rows are excluded from anything that counts or compares
measured band gaps (`measured_properties`), while staying available as what the
source reported.
"""
from __future__ import annotations

from django.db.models import Exists, OuterRef, Q

SUSPECT = "suspect"

#: Device objections that are not an error but put the row outside the single-cell population:
#: a PerovskiteDB module whose J-V the source did not recalculate per cell. Its numbers stay
#: stored and shown, marked as a module, and every selection that excludes a suspect device
#: excludes it too (champion, filters, rankings, statistics, the agent's evidence).
NOT_SINGLE_CELL_CODES = frozenset({"module_jv_not_per_cell"})


def excluding_device_issue_q() -> Q:
    """A DeviceQualityIssue that takes its device out of single-cell selections."""
    return Q(severity=SUSPECT) | Q(code__in=NOT_SINGLE_CELL_CODES)


def excludes_device(issue) -> bool:
    """`excluding_device_issue_q` for one loaded issue."""
    return issue.severity == SUSPECT or issue.code in NOT_SINGLE_CELL_CODES

#: The PerovskiteDB band-gap basis (`Property.context.method_basis`) of a gap assigned from the
#: composition.
COMPOSITION_BASIS = "Composition"

#: PerovskiteDB estimation bases that are a measurement on the sample, shared by the agent's
#: evidence classes (`rag.evidence_class_q`, `rag.evidence_kind_of`) and the device-physics
#: checks. "unstated" and "Literature" say nothing about how the number was obtained;
#: "Composition" is assigned from the formula.
MEASURED_GAP_BASES = frozenset({"Absorption", "EQE", "Absorption Tauc-plot", "UPS", "Absorption Tauc-plot | UPS",
                                "PL"})


def _suspect_property_issue():
    from materials.models import PropertyQualityIssue
    return PropertyQualityIssue.objects.filter(
        property_id=OuterRef("pk"), severity=SUSPECT)


def _suspect_device_issue():
    from materials.models import DeviceQualityIssue
    return DeviceQualityIssue.objects.filter(excluding_device_issue_q(), device_id=OuterRef("pk"))


def eligible_properties(qs):
    """`qs` without properties that carry a suspect objection.

    `Exists`, not a join: a property with two objections would otherwise be
    counted twice by any aggregate built on the result.
    """
    return qs.filter(~Exists(_suspect_property_issue()))


def suspect_properties(qs):
    return qs.filter(Exists(_suspect_property_issue()))


def measured_properties(qs):
    """Eligible rows that are measurements, not composition-assigned values."""
    # Containment, not a key comparison: a key lookup is NULL on a row that states no basis, and
    # NOT NULL would drop that row too.
    return eligible_properties(qs).exclude(context__contains={"method_basis": COMPOSITION_BASIS})


def eligible_devices(qs):
    return qs.filter(~Exists(_suspect_device_issue()))


def material_ids_with(prop_qs):
    """Material ids that have at least one eligible row in `prop_qs`.

    For material filters on a property condition. Written as a subquery on
    the property side: a negated condition across the
    ``properties`` relation inside ``Material.objects.filter(...)`` compiles
    to "no suspect property at all", which drops a material for an objection
    to an unrelated value.
    """
    return eligible_properties(prop_qs).values("material_id")


def _issues(obj):
    """Issues of a row, from the prefetch cache when one exists."""
    cache = getattr(obj, "_prefetched_objects_cache", {}) or {}
    if "quality_issues" in cache:
        return list(cache["quality_issues"])
    return list(obj.quality_issues.all())


def is_property_eligible(prop) -> bool:
    return not any(i.severity == SUSPECT for i in _issues(prop))


def is_device_eligible(device) -> bool:
    return not any(excludes_device(i) for i in _issues(device))
