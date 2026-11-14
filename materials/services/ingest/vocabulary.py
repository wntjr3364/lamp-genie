"""How a stored property name is shown to a reader."""
from __future__ import annotations

#: How a property name is shown where the bare name would mislead. The stored
#: name stays the identifier (queries, routing, aliases); only the label changes.
#:
#: ``stokes_shift`` holds the curated perovskite sheet's "Stokes Shift" column,
#: which is the PL peak minus the absorption onset in nm and is negative where
#: the PL lies above the onset. A Stokes shift is measured against the
#: absorption peak and is positive by construction, so the name alone
#: mislabels the quantity.
DISPLAY_LABELS: dict[str, str] = {
    "stokes_shift": "PL − absorption onset",
}


def display_label(name: str) -> str:
    """Return the label shown for a stored property name (the name itself if none)."""
    return DISPLAY_LABELS.get(name, name)
