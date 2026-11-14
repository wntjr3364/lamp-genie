"""Validate the public data deposit and load it into an empty database of this package's schema.

Validation (always, and alone with `--validate-only`, which opens no database connection):
  * MANIFEST.json names this format and version and lists exactly the thirteen deposited tables;
  * every listed file is present and no CSV in the directory is unlisted;
  * SHA256SUMS covers MANIFEST.json and every listed file, and every digest matches it and
    the manifest;
  * each CSV header equals the manifest's column list, which equals the columns this code loads;
  * each CSV holds the number of records the manifest states;
  * each derived `suspect` column (property, device) is true exactly for the rows that have a
    quality issue of severity suspect in the bundle's issue table, and false otherwise.

Loading: refuses unless the thirteen tables are empty; then, inside one transaction, loads them in
dependency order, the issue tables after their parents (each file COPYed into a typed staging
table, then inserted). A derived column is staged and checked, never inserted. Columns the deposit
leaves out are filled as the schema fills them: the timestamps by the database (the import time),
a nullable column stays NULL, a NOT NULL column takes its model default (an empty text or
container), and `build_run_id` and `detected_by_run_id` point at one BuildRun row the import
creates to record itself. Some are derived (`DERIVED_COLUMNS`): a scan's data source is its
device's, a property without a value gets the reason the schema requires, and an issue's `rule_id`
takes its code with `rule_version` 0 (the deposit holds no rule identifier; one issue per code and
row keeps the schema's key). The id sequences are moved past the loaded ids.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction
from django.db.models.fields import NOT_PROVIDED
from django.utils import timezone

from materials.build.models import (
    BuildRun,
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
from materials.deposit_format import (
    EXPORT_MODELS,
    FORMAT_NAME,
    FORMAT_VERSION,
    SUSPECT_ISSUES,
    derived_columns,
    dropped_columns,
    file_name,
    public_columns,
    stored_columns,
)

#: Parents before children; each issue table after the rows it objects to.
IMPORT_ORDER = (Material, Sample, Device, DeviceLayer, DeviceJVScan, Structure, CrystalSite, Curve, Process,
                Property, PropertyQualityIssue, DeviceQualityIssue, SampleQualityIssue)
assert set(IMPORT_ORDER) == set(EXPORT_MODELS)

#: Columns that point at the BuildRun the import creates.
RUN_COLUMNS = ("build_run_id", "detected_by_run_id")

#: Left-out columns restored on import:
#: (table, column) -> SQL expression over `deposit_stage`.
DERIVED_COLUMNS = {
    ("corpus_device_jv_scan", "data_source"):
        "(SELECT device.data_source FROM corpus_device AS device WHERE device.id = deposit_stage.device_id)",
    # the schema requires a reason on a row without a value
    ("corpus_property", "null_reason"):
        "CASE WHEN deposit_stage.value IS NOT NULL THEN '' WHEN deposit_stage.context ? 'series' "
        "THEN 'series_stored_in_context' ELSE 'no_value_in_source' END",
    # the deposit holds no rule identifier; the code names the rule, version 0 names no version
    ("corpus_property_quality_issue", "rule_id"): "deposit_stage.code",
    ("corpus_property_quality_issue", "rule_version"): "0",
    ("corpus_device_quality_issue", "rule_id"): "deposit_stage.code",
    ("corpus_device_quality_issue", "rule_version"): "0",
    ("corpus_sample_quality_issue", "rule_id"): "deposit_stage.code",
}

csv.field_size_limit(2**31 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(bundle: Path) -> dict:
    path = bundle / "MANIFEST.json"
    if not path.is_file():
        raise CommandError(f"{bundle} has no MANIFEST.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_NAME or manifest.get("format_version") != FORMAT_VERSION:
        raise CommandError(f"MANIFEST.json declares {manifest.get('format')!r} version "
                           f"{manifest.get('format_version')!r}; this importer reads {FORMAT_NAME!r} version "
                           f"{FORMAT_VERSION}.")
    return manifest


def validate_bundle(bundle: Path, *, log=lambda message: None) -> dict:
    """Run every check that needs no database and return the manifest entry of each table."""
    manifest = read_manifest(bundle)
    entries = {e.get("file"): e for e in manifest.get("files", [])}
    expected = {file_name(m): m for m in EXPORT_MODELS}
    if set(entries) != set(expected):
        raise CommandError(f"MANIFEST.json lists {sorted(entries)}; the deposit has {sorted(expected)}.")
    missing = sorted(name for name in entries if not (bundle / name).is_file())
    if missing:
        raise CommandError(f"Files listed in MANIFEST.json are missing: {', '.join(missing)}")
    unlisted = sorted(p.name for p in bundle.iterdir()
                      if p.name.endswith((".csv", ".csv.gz")) and p.name not in entries)
    if unlisted:
        raise CommandError(f"The bundle holds unlisted CSV files: {', '.join(unlisted)}")

    sums_path = bundle / "SHA256SUMS"
    if not sums_path.is_file():
        raise CommandError(f"{bundle} has no SHA256SUMS")
    sums = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        sums[name] = digest
    for name in sorted(set(entries) | {"MANIFEST.json"}):
        if name not in sums:
            raise CommandError(f"Snapshot checksum mismatch: SHA256SUMS does not cover {name}")
        actual = _sha256(bundle / name)
        if actual != sums[name] or (name in entries and actual != entries[name].get("sha256")):
            raise CommandError(f"Snapshot checksum mismatch: {name} has sha256 {actual}")

    tables = {}
    flagged = {}   # table -> ids whose `suspect` is true
    objected = {}  # issue table -> ids of the rows with a suspect-severity issue
    suspect_tables = {issues: table for table, (issues, _) in SUSPECT_ISSUES.items()}
    for name, model in expected.items():
        entry = entries[name]
        table = model._meta.db_table
        columns = public_columns(model)
        if entry.get("table") != table or entry.get("columns") != columns:
            raise CommandError(f"{name}: the manifest's table or column list differs from the deposited columns "
                               f"of {table}")
        flag = columns.index("suspect") if "suspect" in derived_columns(model) else None
        owner = (columns.index(SUSPECT_ISSUES[suspect_tables[table]][1]), columns.index("severity")) \
            if table in suspect_tables else None
        with gzip.open(bundle / name, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != columns:
                raise CommandError(f"{name}: the CSV header differs from the column list {columns}")
            rows = 0
            for record in reader:
                if len(record) != len(columns):
                    raise CommandError(f"{name}: record {rows + 1} has {len(record)} fields, not {len(columns)}")
                rows += 1
                if flag is not None:
                    if record[flag] == "true":
                        flagged.setdefault(table, set()).add(record[0])
                    elif record[flag] != "false":
                        raise CommandError(f"{name}: record {rows} has suspect {record[flag]!r}, not true or false")
                if owner is not None and record[owner[1]] == "suspect":
                    objected.setdefault(table, set()).add(record[owner[0]])
        if rows != entry.get("rows"):
            raise CommandError(f"CSV row count differs: {name} holds {rows:,} records, MANIFEST.json says "
                               f"{entry.get('rows')!r}")
        log(f"  {name}: {rows:,} records, sha256 and columns match")
        tables[model] = entry
    for table, (issues, _) in SUSPECT_ISSUES.items():
        marked, expected_ids = flagged.get(table, set()), objected.get(issues, set())
        if marked != expected_ids:
            raise CommandError(f"{table}.suspect disagrees with {issues}: {len(marked - expected_ids):,} rows are "
                               f"true without a suspect issue, {len(expected_ids - marked):,} rows with one are "
                               f"false")
        log(f"  {table}.suspect: true in {len(marked):,} rows, as {issues} states")
    return tables


def _fill_values(model, run_id: int) -> dict:
    """Return column -> value for each left-out column that the database does not fill itself."""
    fill = {}
    for column in dropped_columns(model):
        field = next(f for f in model._meta.concrete_fields if f.column == column)
        if (model._meta.db_table, column) in DERIVED_COLUMNS:
            continue
        if column in RUN_COLUMNS:
            fill[column] = run_id
        elif field.null or field.db_default is not NOT_PROVIDED:
            continue  # NULL, or the database default (the timestamps: the import time)
        else:
            fill[column] = field.get_db_prep_save(field.get_default(), connection)
    return fill


class Command(BaseCommand):
    help = "Validate the public data deposit and load it into an empty database."

    def add_arguments(self, parser):
        parser.add_argument("--input", required=True, help="the deposit directory")
        parser.add_argument("--validate-only", action="store_true", help="check the bundle and write nothing")

    def handle(self, *args, **options):
        bundle = Path(options["input"]).expanduser().resolve()
        if not bundle.is_dir():
            raise CommandError(f"{bundle} is not a directory")
        tables = validate_bundle(bundle, log=self.stdout.write)
        if options["validate_only"]:
            self.stdout.write(self.style.SUCCESS(f"Bundle valid: {len(tables)} tables, nothing written"))
            return
        if connection.vendor != "postgresql":
            raise CommandError(f"The deposit is loaded into PostgreSQL, not {connection.vendor!r}.")
        quote = connection.ops.quote_name
        with transaction.atomic():
            with connection.cursor() as cursor:
                present = set(connection.introspection.table_names(cursor))
                absent = [m._meta.db_table for m in IMPORT_ORDER if m._meta.db_table not in present]
                if absent:
                    raise CommandError(f"The database lacks the corpus tables {absent}; run migrate first.")
                for model in IMPORT_ORDER:
                    cursor.execute(f"LOCK TABLE {quote(model._meta.db_table)} IN ACCESS EXCLUSIVE MODE")
                occupied = []
                for model in IMPORT_ORDER:
                    cursor.execute(f"SELECT EXISTS (SELECT 1 FROM {quote(model._meta.db_table)})")
                    if cursor.fetchone()[0]:
                        occupied.append(model._meta.db_table)
                if occupied:
                    raise CommandError(f"Refusing to import into non-empty public tables: {', '.join(occupied)}")
                manifest_digest = _sha256(bundle / "MANIFEST.json")
                run = BuildRun.objects.create(
                    target_db=connection.settings_dict.get("NAME") or "", git_head="",
                    inputs_digest=manifest_digest, environment_digest="", contract_digest="",
                    counts={m._meta.db_table: tables[m]["rows"] for m in IMPORT_ORDER},
                    status="completed", finished_at=timezone.now())
                for model in IMPORT_ORDER:
                    self._load(cursor, bundle, model, tables[model], run.id)
                for model in IMPORT_ORDER:
                    table = model._meta.db_table
                    cursor.execute(f"SELECT setval(pg_get_serial_sequence(%s, 'id'), COALESCE(MAX(id), 1), "
                                   f"MAX(id) IS NOT NULL) FROM {quote(table)}", [table])
        self.stdout.write(self.style.SUCCESS(
            f"Imported {sum(e['rows'] for e in tables.values()):,} rows into {len(tables)} tables"))

    def _load(self, cursor, bundle: Path, model, entry: dict, run_id: int) -> None:
        quote = connection.ops.quote_name
        table = model._meta.db_table
        file_columns = ", ".join(quote(c) for c in entry["columns"])
        column_sql = ", ".join(quote(c) for c in stored_columns(model))
        cursor.execute(f"CREATE TEMP TABLE deposit_stage AS SELECT {column_sql} FROM {quote(table)} WITH NO DATA")
        for column in derived_columns(model):  # staged, checked by validate_bundle, never inserted
            cursor.execute(f"ALTER TABLE deposit_stage ADD COLUMN {quote(column)} boolean")
        with gzip.open(bundle / entry["file"], "rb") as handle:
            cursor.copy_expert(f"COPY deposit_stage ({file_columns}) FROM STDIN WITH (FORMAT csv, HEADER true)",
                               handle)
        fill = _fill_values(model, run_id)
        derived = {c: sql for (t, c), sql in DERIVED_COLUMNS.items() if t == table}
        target = column_sql + "".join(f", {quote(c)}" for c in (*fill, *derived))
        source = column_sql + "".join(", %s" for _ in fill) + "".join(f", {sql}" for sql in derived.values())
        cursor.execute(f"INSERT INTO {quote(table)} ({target}) SELECT {source} FROM deposit_stage ORDER BY id",
                       list(fill.values()))
        loaded = cursor.rowcount
        cursor.execute("DROP TABLE deposit_stage")
        if loaded != entry["rows"]:
            raise CommandError(f"CSV row count differs: {table} loaded {loaded:,} rows, MANIFEST.json says "
                               f"{entry['rows']:,}")
        self.stdout.write(f"  {table}: {loaded:,} rows")
