"""Constraint triggers for the ownership rules of the corpus schema, and the rules a plain
constraint cannot state (source keys unique per data source when the source is reached through a
link, no Sample of a computed source, a Process keyed under its Sample, a stored data_source equal
to its parent's, keys that never change).

Parent equality is null-safe (IS DISTINCT FROM): NULL is a value, not a wildcard. The triggers are
DEFERRABLE INITIALLY IMMEDIATE: a row is checked when its statement ends, or at commit / SET
CONSTRAINTS IMMEDIATE when deferred. A deferred event still carries the row version of its
statement, so every check reads the row as it is when the check runs (a two-statement repair is
judged on its final state) and skips a row deleted since. Each refusal raises check_violation (an
IntegrityError in Django) naming the rule it enforces. A row is checked when it is written and when
a row it depends on is updated; keys (Material data_source / external_id, every source_key and
data_source column) never change, so a parent's source or key cannot move under its children (a
rekey is a new row).
"""
from django.db import migrations

COMPUTED = "'MP', 'OQMD', 'JARVIS', 'JARVIS-SC', '2DMatPedia', 'C2DB', 'QMOF', 'HCEP'"

FUNCTIONS = r"""
CREATE OR REPLACE FUNCTION corpus_refuse(rule text, detail text) RETURNS void AS $$
BEGIN
    RAISE EXCEPTION 'ownership rule violated: % (%)', rule, detail USING ERRCODE = 'check_violation';
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION corpus_keys_frozen() RETURNS trigger AS $$
DECLARE col text;
BEGIN
    FOREACH col IN ARRAY TG_ARGV LOOP
        IF to_jsonb(NEW) -> col IS DISTINCT FROM to_jsonb(OLD) -> col THEN
            PERFORM corpus_refuse('keys never change (a rekey is a new row)',
                                  TG_TABLE_NAME || '.' || col || ' ' || coalesce(to_jsonb(OLD) ->> col, ''));
        END IF;
    END LOOP;
    RETURN NULL;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION corpus_check_sample() RETURNS trigger AS $$
DECLARE r corpus_sample; ds text;
BEGIN
    SELECT * INTO r FROM corpus_sample WHERE id = NEW.id;
    IF NOT FOUND THEN RETURN NULL; END IF;
    SELECT data_source INTO ds FROM corpus_material WHERE id = r.material_id;
    IF ds IN (%(computed)s) THEN
        PERFORM corpus_refuse('computed sources create no Sample', r.source_key);
    END IF;
    IF EXISTS (SELECT 1 FROM corpus_sample s JOIN corpus_material m ON m.id = s.material_id
               WHERE s.source_key = r.source_key AND s.id <> r.id AND m.data_source = ds) THEN
        PERFORM corpus_refuse('unique (data_source, source_key) on Sample', r.source_key);
    END IF;
    IF TG_OP = 'UPDATE' AND (
           EXISTS (SELECT 1 FROM corpus_device WHERE sample_id = r.id AND material_id <> r.material_id)
        OR EXISTS (SELECT 1 FROM corpus_property WHERE sample_id = r.id AND material_id <> r.material_id)
        OR EXISTS (SELECT 1 FROM corpus_process WHERE sample_id = r.id AND material_id <> r.material_id)
        OR EXISTS (SELECT 1 FROM corpus_structure WHERE sample_id = r.id AND material_id <> r.material_id)) THEN
        PERFORM corpus_refuse('same-Material: children of a Sample belong to its Material', r.source_key);
    END IF;
    RETURN NULL;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION corpus_check_device() RETURNS trigger AS $$
DECLARE r corpus_device;
BEGIN
    SELECT * INTO r FROM corpus_device WHERE id = NEW.id;
    IF NOT FOUND THEN RETURN NULL; END IF;
    IF r.data_source IS DISTINCT FROM (SELECT data_source FROM corpus_material WHERE id = r.material_id) THEN
        PERFORM corpus_refuse('a stored data_source is the source of its material', r.source_key);
    END IF;
    IF (SELECT material_id FROM corpus_sample WHERE id = r.sample_id) IS DISTINCT FROM r.material_id THEN
        PERFORM corpus_refuse('Device.material == Device.sample.material', r.source_key);
    END IF;
    IF TG_OP = 'UPDATE' THEN
        IF EXISTS (SELECT 1 FROM corpus_property WHERE device_id = r.id
                   AND (sample_id IS DISTINCT FROM r.sample_id OR material_id <> r.material_id)) THEN
            PERFORM corpus_refuse('Property.device.sample == Property.sample', r.source_key);
        END IF;
        IF EXISTS (SELECT 1 FROM corpus_process WHERE device_id = r.id AND sample_id IS DISTINCT FROM r.sample_id) THEN
            PERFORM corpus_refuse('Process.device.sample == Process.sample', r.source_key);
        END IF;
        IF EXISTS (SELECT 1 FROM corpus_structure st JOIN corpus_device_layer l ON l.id = st.layer_id
                   WHERE l.device_id = r.id AND st.sample_id IS DISTINCT FROM r.sample_id) THEN
            PERFORM corpus_refuse('Structure.layer.device.sample == Structure.sample', r.source_key);
        END IF;
    END IF;
    RETURN NULL;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION corpus_check_jv_scan() RETURNS trigger AS $$
DECLARE r corpus_device_jv_scan;
BEGIN
    SELECT * INTO r FROM corpus_device_jv_scan WHERE id = NEW.id;
    IF NOT FOUND THEN RETURN NULL; END IF;
    IF r.data_source IS DISTINCT FROM (SELECT data_source FROM corpus_device WHERE id = r.device_id) THEN
        PERFORM corpus_refuse('a stored data_source is the source of its device', r.source_key);
    END IF;
    RETURN NULL;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION corpus_check_device_layer() RETURNS trigger AS $$
DECLARE r corpus_device_layer;
BEGIN
    SELECT * INTO r FROM corpus_device_layer WHERE id = NEW.id;
    IF NOT FOUND OR TG_OP <> 'UPDATE' THEN RETURN NULL; END IF;
    IF EXISTS (SELECT 1 FROM corpus_structure WHERE layer_id = r.id
               AND sample_id IS DISTINCT FROM (SELECT sample_id FROM corpus_device WHERE id = r.device_id)) THEN
        PERFORM corpus_refuse('Structure.layer.device.sample == Structure.sample', r.source_key);
    END IF;
    IF EXISTS (SELECT 1 FROM corpus_property p JOIN corpus_structure st ON st.id = p.structure_id
               WHERE st.layer_id = r.id AND p.device_id IS NOT NULL AND p.device_id <> r.device_id) THEN
        PERFORM corpus_refuse('Property.device == Structure.layer.device', r.source_key);
    END IF;
    RETURN NULL;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION corpus_check_structure() RETURNS trigger AS $$
DECLARE r corpus_structure; layer_sample bigint; layer_device bigint;
BEGIN
    SELECT * INTO r FROM corpus_structure WHERE id = NEW.id;
    IF NOT FOUND THEN RETURN NULL; END IF;
    IF r.data_source IS DISTINCT FROM (SELECT data_source FROM corpus_material WHERE id = r.material_id) THEN
        PERFORM corpus_refuse('a stored data_source is the source of its material', r.source_key);
    END IF;
    IF r.sample_id IS NOT NULL
       AND (SELECT material_id FROM corpus_sample WHERE id = r.sample_id) IS DISTINCT FROM r.material_id THEN
        PERFORM corpus_refuse('Structure.material == Structure.sample.material', r.source_key);
    END IF;
    IF r.layer_id IS NOT NULL THEN
        SELECT d.sample_id, d.id INTO layer_sample, layer_device
          FROM corpus_device_layer l JOIN corpus_device d ON d.id = l.device_id WHERE l.id = r.layer_id;
        IF layer_sample IS DISTINCT FROM r.sample_id THEN
            PERFORM corpus_refuse('Structure.layer.device.sample == Structure.sample', r.source_key);
        END IF;
    END IF;
    IF TG_OP = 'UPDATE' THEN
        IF EXISTS (SELECT 1 FROM corpus_property WHERE structure_id = r.id
                   AND (sample_id IS DISTINCT FROM r.sample_id OR material_id <> r.material_id)) THEN
            PERFORM corpus_refuse('Property.structure.sample == Property.sample', r.source_key);
        END IF;
        IF r.layer_id IS NOT NULL AND EXISTS (SELECT 1 FROM corpus_property WHERE structure_id = r.id
                                              AND device_id IS NOT NULL AND device_id <> layer_device) THEN
            PERFORM corpus_refuse('Property.device == Structure.layer.device', r.source_key);
        END IF;
    END IF;
    RETURN NULL;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION corpus_check_curve() RETURNS trigger AS $$
DECLARE r corpus_curve; ds text;
BEGIN
    SELECT * INTO r FROM corpus_curve WHERE id = NEW.id;
    IF NOT FOUND THEN RETURN NULL; END IF;
    SELECT m.data_source INTO ds FROM corpus_material m
     WHERE m.id = COALESCE(r.material_id, (SELECT material_id FROM corpus_sample WHERE id = r.sample_id));
    IF EXISTS (SELECT 1 FROM corpus_curve c LEFT JOIN corpus_sample s ON s.id = c.sample_id
               JOIN corpus_material m ON m.id = COALESCE(c.material_id, s.material_id)
               WHERE c.source_key = r.source_key AND c.id <> r.id AND m.data_source = ds) THEN
        PERFORM corpus_refuse('unique (data_source, source_key) on Curve', r.source_key);
    END IF;
    IF TG_OP = 'UPDATE' AND EXISTS (SELECT 1 FROM corpus_property WHERE curve_id = r.id
                                    AND (sample_id IS DISTINCT FROM r.sample_id
                                         OR r.material_id IS NOT NULL AND material_id <> r.material_id)) THEN
        PERFORM corpus_refuse('Property.curve.sample == Property.sample', r.source_key);
    END IF;
    RETURN NULL;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION corpus_check_process() RETURNS trigger AS $$
DECLARE r corpus_process; s_key text; s_material bigint;
BEGIN
    SELECT * INTO r FROM corpus_process WHERE id = NEW.id;
    IF NOT FOUND THEN RETURN NULL; END IF;
    IF r.data_source IS DISTINCT FROM (SELECT data_source FROM corpus_material WHERE id = r.material_id) THEN
        PERFORM corpus_refuse('a stored data_source is the source of its material', r.source_key);
    END IF;
    SELECT source_key, material_id INTO s_key, s_material FROM corpus_sample WHERE id = r.sample_id;
    IF s_material IS DISTINCT FROM r.material_id THEN
        PERFORM corpus_refuse('Process.material == Process.sample.material', r.source_key);
    END IF;
    IF r.device_id IS NOT NULL
       AND (SELECT sample_id FROM corpus_device WHERE id = r.device_id) IS DISTINCT FROM r.sample_id THEN
        PERFORM corpus_refuse('Process.device.sample == Process.sample', r.source_key);
    END IF;
    IF left(r.source_key, length(s_key) + 1) IS DISTINCT FROM s_key || '|' THEN
        PERFORM corpus_refuse('a Process belongs to the Sample its source_key names', r.source_key);
    END IF;
    RETURN NULL;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION corpus_check_property() RETURNS trigger AS $$
DECLARE r corpus_property; o_sample bigint; o_material bigint; o_layer bigint; o_device bigint;
BEGIN
    SELECT * INTO r FROM corpus_property WHERE id = NEW.id;
    IF NOT FOUND THEN RETURN NULL; END IF;
    IF r.data_source IS DISTINCT FROM (SELECT data_source FROM corpus_material WHERE id = r.material_id) THEN
        PERFORM corpus_refuse('a stored data_source is the source of its material', r.source_key);
    END IF;
    IF r.sample_id IS NOT NULL
       AND (SELECT material_id FROM corpus_sample WHERE id = r.sample_id) IS DISTINCT FROM r.material_id THEN
        PERFORM corpus_refuse('Property.material == Property.sample.material', r.source_key);
    END IF;
    IF r.device_id IS NOT NULL THEN
        SELECT sample_id, material_id INTO o_sample, o_material FROM corpus_device WHERE id = r.device_id;
        IF o_sample IS DISTINCT FROM r.sample_id OR o_material IS DISTINCT FROM r.material_id THEN
            PERFORM corpus_refuse('Property.device.sample == Property.sample', r.source_key);
        END IF;
    END IF;
    IF r.curve_id IS NOT NULL THEN
        SELECT sample_id, material_id INTO o_sample, o_material FROM corpus_curve WHERE id = r.curve_id;
        IF o_sample IS DISTINCT FROM r.sample_id THEN
            PERFORM corpus_refuse('Property.curve.sample == Property.sample', r.source_key);
        END IF;
        IF o_material IS NOT NULL AND o_material <> r.material_id THEN
            PERFORM corpus_refuse('Property.material == Curve.material', r.source_key);
        END IF;
    END IF;
    IF r.structure_id IS NOT NULL THEN
        SELECT sample_id, material_id, layer_id INTO o_sample, o_material, o_layer
          FROM corpus_structure WHERE id = r.structure_id;
        IF o_sample IS DISTINCT FROM r.sample_id THEN
            PERFORM corpus_refuse('Property.structure.sample == Property.sample', r.source_key);
        END IF;
        IF o_material IS DISTINCT FROM r.material_id THEN
            PERFORM corpus_refuse('Property.material == Property.structure.material', r.source_key);
        END IF;
        IF o_layer IS NOT NULL AND r.device_id IS NOT NULL THEN
            SELECT device_id INTO o_device FROM corpus_device_layer WHERE id = o_layer;
            IF o_device IS DISTINCT FROM r.device_id THEN
                PERFORM corpus_refuse('Property.device == Structure.layer.device', r.source_key);
            END IF;
        END IF;
    END IF;
    RETURN NULL;
END $$ LANGUAGE plpgsql;
""".replace("%(computed)s", COMPUTED)

TRIGGERS = [
    ("corpus_sample", "corpus_check_sample"),
    ("corpus_device", "corpus_check_device"),
    ("corpus_device_layer", "corpus_check_device_layer"),
    ("corpus_device_jv_scan", "corpus_check_jv_scan"),
    ("corpus_structure", "corpus_check_structure"),
    ("corpus_curve", "corpus_check_curve"),
    ("corpus_process", "corpus_check_process"),
    ("corpus_property", "corpus_check_property"),
]
#: table -> key columns that never change after insertion
FROZEN_KEYS = [
    ("corpus_material", ("data_source", "external_id")),
    ("corpus_sample", ("source_key",)),
    ("corpus_device", ("data_source", "source_key")),
    ("corpus_device_layer", ("source_key",)),
    ("corpus_device_jv_scan", ("data_source", "source_key")),
    ("corpus_structure", ("data_source", "source_key")),
    ("corpus_curve", ("source_key",)),
    ("corpus_process", ("data_source", "source_key")),
    ("corpus_property", ("data_source", "source_key")),
]

CREATE = FUNCTIONS + "\n".join(
    [f"CREATE CONSTRAINT TRIGGER {fn}_trg AFTER INSERT OR UPDATE ON {table} "
     f"DEFERRABLE INITIALLY IMMEDIATE FOR EACH ROW EXECUTE FUNCTION {fn}();" for table, fn in TRIGGERS]
    + [f"CREATE CONSTRAINT TRIGGER {table}_keys_trg AFTER UPDATE ON {table} DEFERRABLE INITIALLY IMMEDIATE "
       f"FOR EACH ROW EXECUTE FUNCTION corpus_keys_frozen({', '.join(repr(c) for c in cols)});"
       for table, cols in FROZEN_KEYS])

DROP = "\n".join(
    [f"DROP TRIGGER IF EXISTS {fn}_trg ON {table};" for table, fn in TRIGGERS]
    + [f"DROP TRIGGER IF EXISTS {table}_keys_trg ON {table};" for table, _ in FROZEN_KEYS]
    + [f"DROP FUNCTION IF EXISTS {fn}();" for _, fn in TRIGGERS]
    + ["DROP FUNCTION IF EXISTS corpus_keys_frozen();", "DROP FUNCTION IF EXISTS corpus_refuse(text, text);"])


def forwards(apps, schema_editor):
    if schema_editor.connection.vendor == "postgresql":
        schema_editor.execute(CREATE, params=None)


def backwards(apps, schema_editor):
    if schema_editor.connection.vendor == "postgresql":
        schema_editor.execute(DROP, params=None)


class Migration(migrations.Migration):

    dependencies = [("corpus", "0001_initial")]

    operations = [migrations.RunPython(forwards, backwards)]
