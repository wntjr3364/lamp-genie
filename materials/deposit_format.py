"""File format of the public data deposit: the deposited tables and the columns each leaves out."""
from __future__ import annotations

from materials.build.models import (
    CrystalSite,
    Curve,
    Device,
    DeviceJVScan,
    DeviceLayer,
    DeviceQualityIssue,
    Material,
    Process,
    Property,
    PropertyQualityIssue,
    Sample,
    SampleQualityIssue,
    Structure,
)


FORMAT_NAME = "lamp-public-database"
FORMAT_VERSION = 2

#: The deposited tables, in the order the manifest lists them.
EXPORT_MODELS = (Material, Property, Structure, CrystalSite, Sample, Curve, Process, Device, DeviceLayer,
                 DeviceJVScan, PropertyQualityIssue, DeviceQualityIssue, SampleQualityIssue)


#: Dropped from every table: timestamps, the user who created a row, and the import that wrote it.
VOLATILE_COLUMNS = ("created_at", "updated_at", "created_by_id", "build_run_id", "detected_by_run_id")

#: Empty (NULL or "") in every row of the database.
EMPTY_COLUMNS = {
    "corpus_material": ("parent_compound",),
    "corpus_structure": ("cif_file", "xyz_file", "mol_file", "pdb_file"),
    "corpus_sample": ("cited_work",),
    "corpus_crystal_site": ("coordination",),
    "corpus_process": ("spin_speed_rpm", "spin_time_s"),
}

#: Columns whose content records how a row was derived, not a value of the source.
INTERNAL_COLUMNS = {
    "corpus_material": ("classification_basis",),
    "corpus_structure": ("ingestion",),
    "corpus_device": ("ff_rule_basis",),
    "corpus_device_jv_scan": ("quality_context",),
}

#: Not part of the deposited values: curation status, remarks, the value as written before unit
#: conversion, constants, near-empty columns, and columns another column restates. Of a quality
#: issue: the rule text, rule identifier and version, assumptions, and note, which the agent does
#: not read.
OMITTED_COLUMNS = {
    "corpus_material": ("composition_status", "composition_group"),
    "corpus_property": ("raw_value", "null_reason", "note"),
    "corpus_structure": ("symprec", "note"),
    "corpus_crystal_site": ("note",),
    "corpus_curve": ("status_reason", "duplicate_of_id", "duplicate_candidates"),
    "corpus_process": ("step_count", "layer_index"),
    "corpus_device": ("ff_raw_value", "ff_raw_unit", "note"),
    "corpus_device_layer": ("note",),
    "corpus_device_jv_scan": ("data_source",),
    "corpus_property_quality_issue": ("rule", "rule_id", "rule_version", "assumptions", "note"),
    "corpus_device_quality_issue": ("rule", "rule_id", "rule_version", "assumptions", "note"),
    "corpus_sample_quality_issue": ("rule_id", "assumptions", "note"),
}

#: Columns the database does not store, written after the stored columns:
#: table -> {column: SQL boolean over the table's own row}.
#: `suspect`: the row has at least one quality issue of severity suspect.
DERIVED_EXPORT_COLUMNS = {
    "corpus_property": {"suspect": "EXISTS (SELECT 1 FROM corpus_property_quality_issue AS issue "
                                   "WHERE issue.property_id = corpus_property.id AND issue.severity = 'suspect')"},
    "corpus_device": {"suspect": "EXISTS (SELECT 1 FROM corpus_device_quality_issue AS issue "
                                 "WHERE issue.device_id = corpus_device.id AND issue.severity = 'suspect')"},
}
#: For each derived flag: the issue table the flag is checked against and its foreign-key column.
SUSPECT_ISSUES = {"corpus_property": ("corpus_property_quality_issue", "property_id"),
                  "corpus_device": ("corpus_device_quality_issue", "device_id")}


def file_name(model) -> str:
    return model._meta.db_table.removeprefix("corpus_") + ".csv.gz"


def dropped_columns(model) -> dict[str, str]:
    """column -> why it is not deposited."""
    table = model._meta.db_table
    columns = {f.column for f in model._meta.concrete_fields}
    out = {c: "volatile" for c in VOLATILE_COLUMNS if c in columns}
    out.update({c: "empty in every row" for c in EMPTY_COLUMNS.get(table, ())})
    out.update({c: "build-internal content" for c in INTERNAL_COLUMNS.get(table, ())})
    out.update({c: "not a deposited value" for c in OMITTED_COLUMNS.get(table, ())})
    return out


def stored_columns(model) -> list[str]:
    """The deposited columns of `model` that the database stores, in table order."""
    dropped = dropped_columns(model)
    return [f.column for f in model._meta.concrete_fields if f.column not in dropped]


def derived_columns(model) -> list[str]:
    """The deposited columns of `model` that the database does not store."""
    return list(DERIVED_EXPORT_COLUMNS.get(model._meta.db_table, {}))


def public_columns(model) -> list[str]:
    """The deposited columns of `model`: the stored ones in table order, then the derived ones."""
    return stored_columns(model) + derived_columns(model)
