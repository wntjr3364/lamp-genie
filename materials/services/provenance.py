"""One definition of "a record the platform publishes as its own".

`Property`, `Device`, `Structure` and `Process` carry a nullable `created_by`
FK. NULL means the row belongs to the corpus (every row the deposit loads);
non-NULL means a user added it (the deposit holds no such row).

A read that does not filter on provenance would include such rows. Range
filters, ranking subqueries, presence filters and the agent's tools all use the
filters defined here, so the constraint has a single spelling.

Two shapes are provided, because Django expresses the constraint differently
depending on which side of the relation a query starts from:

  * DIRECT   - you already hold a material and want its rows:
                   `imported_properties(mat).filter(name__iexact="bandgap")`
  * JOIN     - you are filtering/annotating `Material` across the relation:
                   `Material.objects.filter(**imported_property_join(name="x"))`
               or `Q(...) & IMPORTED_DEVICE_JOIN`

Lookups spanning a multi-valued relation constrain the same related row only
when they appear in a single `filter()` call. `imported_property_join()`
therefore returns a dict to merge into the caller's filter kwargs; a separate
`.filter()` call would open a second join and stop constraining that row.

Not used by the nested rows of `MaterialSerializer`, which include rows of any
provenance.
"""
from __future__ import annotations

from django.db.models import Q

__all__ = [
    "IMPORTED_ONLY",
    "IMPORTED_DEVICE_JOIN",
    "IMPORTED_PROPERTY_JOIN",
    "imported_devices",
    "imported_properties",
    "imported_structures",
    "imported_property_join",
    "imported_process_join",
]

#: Lookup for a queryset of `Device`/`Property` rows themselves.
IMPORTED_ONLY = {"created_by__isnull": True}

#: The same constraint expressed across `Material -> <relation>`.
IMPORTED_DEVICE_JOIN = Q(devices__created_by__isnull=True)
IMPORTED_PROPERTY_JOIN = Q(properties__created_by__isnull=True)


def imported_devices(material):
    """The devices of `material` that the platform publishes as its own."""
    return material.devices.filter(**IMPORTED_ONLY)


def imported_structures(material):
    """The structures of `material` that the platform publishes as its own.

    Used by readers that resolve a material by formula (the agent's structure
    tool), so a CIF with a `created_by` user never stands in for the
    material's structure.
    """
    return material.structures.filter(**IMPORTED_ONLY)


def imported_properties(material):
    """The properties of `material` that the platform publishes as its own."""
    return material.properties.filter(**IMPORTED_ONLY)


def _join(relation: str, lookups: dict) -> dict:
    """Prefix each lookup with `relation__` unless it already carries it.

    Accepts lookups relative to the related model (`name__iexact="bandgap"`)
    and `Material`-relative ones (`properties__name="bandgap"`), and adds the
    `created_by__isnull` constraint.
    """
    prefix = f"{relation}__"
    normalised = {
        (key if key.startswith(prefix) else prefix + key): value
        for key, value in lookups.items()
    }
    normalised[f"{prefix}created_by__isnull"] = True
    return normalised


def imported_property_join(**lookups) -> dict:
    """Filter kwargs for `Material`, scoped to imported properties."""
    return _join("properties", lookups)


def imported_process_join(**lookups) -> dict:
    """Filter kwargs for `Material`, scoped to imported synthesis records."""
    return _join("processes", lookups)
