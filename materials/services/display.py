"""Display names for records, where a person or a model reads them.

`Material.name` means different things per source, so it is not printed bare:

* **StarryData2** stores the sample label of a digitized figure in `name` ("1st", "0.47", "2vol%",
  "as-synthesized"). The label follows the formula, with the DOI of the paper that defines it:
  "formula – sample label (DOI ...)". A record whose label equals its formula still carries the DOI.
* **Curated-OLED** stores some emitters under the compound number their paper uses ("2", "3", "8"),
  with a blank formula; these are shown as "Compound 2 (<reference>)".
* **HCEP**, **HOPV** and **Curated-OLED** molecules carry their own name after the formula, since
  different molecules share one formula.
* **PerovskiteDB** and **Curated-Perovskite** records are named with their dimensional context
  ("CsPbBr3 (3D)", "CsPbBr3 (0D)"), which the stored name carries.

Every other record shows the formula, else the name.
"""
from __future__ import annotations

import re

SAMPLE_LABEL_SOURCES = ("StarryData2",)
COMPOUND_NUMBER_SOURCES = ("Curated-OLED",)
MOLECULE_NAME_SOURCES = ("HCEP", "HOPV", "Curated-OLED")
CONTEXT_NAME_SOURCES = ("PerovskiteDB", "Curated-Perovskite")

_DOI_RE = re.compile(r"\b(?:DOI\s+|doi\.org/)(10\.\d{4,9}/[^\s;,]+)", re.IGNORECASE)
_OLED_REF_RE = re.compile(r"Ref:\s*([^;]+)")
_COMPOUND_NUMBER_RE = re.compile(r"\d+[a-z]?")


def doi_of(text: str) -> str:
    """Return the first DOI written in `text` ("DOI 10.x/y" or "doi.org/10.x/y"), or ""."""
    match = _DOI_RE.search(text or "")
    return match.group(1).rstrip(".") if match else ""


def display_label(formula: str, name: str, data_source: str, description: str = "") -> str:
    """Return the record's display name; "" when it has neither formula nor name."""
    formula = (formula or "").strip()
    name = (name or "").strip()
    source = data_source or ""
    if source in SAMPLE_LABEL_SOURCES and formula:
        doi = doi_of(description)
        label = f"{formula} – {name}" if name and name != formula else formula
        return label + (f" (DOI {doi})" if doi else "")
    if source in MOLECULE_NAME_SOURCES and formula and name and name != formula:
        return f"{formula} ({name})"
    if source in CONTEXT_NAME_SOURCES and formula and name.startswith(formula) and name != formula:
        return name
    if source in COMPOUND_NUMBER_SOURCES and not formula and _COMPOUND_NUMBER_RE.fullmatch(name):
        ref = _OLED_REF_RE.search(description or "")
        return f"Compound {name}" + (f" ({ref.group(1).strip()})" if ref else "")
    return formula or name


def material_display_label(material) -> str:
    return display_label(getattr(material, "formula", "") or "",
                         getattr(material, "name", "") or "",
                         getattr(material, "data_source", "") or "",
                         getattr(material, "description", "") or "")
