"""Read-only serializers of the corpus.

Every serializer names its fields: the list is the public allowlist of its model, and a column it
does not name is not published. Never published: `note` columns, `build_run` / `detected_by_run`,
source keys, timestamps, `Material.classification_basis`, `Device.ff_rule_basis`,
`DeviceJVScan.quality_context`, `Structure.ingestion`, and the rule, rule id, rule version,
assumptions and locator of a quality issue. The JSON columns are published through
`materials.public_projection`, never copied.

The corpus is read-only: every field is read-only and no serializer writes.
"""
from typing import Optional

from collections import defaultdict

from django.db import models as dj_models
from django.db.models import Q, prefetch_related_objects
from rest_framework import serializers

from . import public_projection
from .services.device_policy import (
    CHAMPION_POOL_ATTR as _CHAMPION_POOL_ATTR,
    CHAMPION_POOL_SIZE,
    champion_cell,
    is_one_sun_complete_cell,
    one_sun_cell_q,
    pce_basis,
    imported_solar_cells,
    is_plausible_cell,
    pick_champion,
)
from .models import (
    Curve,
    CrystalSite,
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

#: Points a curve preview carries; the full arrays are the curve's `raw_x`/`raw_y`.
CURVE_PREVIEW_POINTS = 50

#: Context key naming the model whose rows are serialized in full: the large JSON arrays (cfid,
#: context.series) are published only on that model's own rows, never on an inline row. `ask` does
#: not set it, so the `materials` records it returns leave them out.
DETAIL_OF = "detail_of"


def _is_detail(field) -> bool:
    serializer = field.parent
    model = getattr(getattr(serializer, "Meta", None), "model", None)
    return model is not None and field.context.get(DETAIL_OF) == model.__name__


class ProjectedJSONField(serializers.Field):
    """A governed JSON column, published through `public_projection.project` (read-only)."""

    def __init__(self, column: str, **kwargs):
        self.column = column
        kwargs["read_only"] = True
        super().__init__(**kwargs)

    def to_representation(self, value):
        return public_projection.project(self.column, value, detail=_is_detail(self))


class _ReadOnly(serializers.ModelSerializer):
    """Every model field read-only; declared fields are read-only by construction."""

    def get_fields(self):
        fields = super().get_fields()
        for field in fields.values():
            field.read_only = True
        return fields


# Curve links, resolved in batches for the serialized window
#: Context caches of one serialization: {curve id: ids of its `duplicate_of` group} and {(data
#: source, curve source key): curve id or None}. A list serializer fills them for its whole window
#: before any row is serialized (three queries per window at most); a row serialized alone fills
#: them for itself.
_CURVE_GROUPS = "_curve_groups"
_CURVE_KEYS = "_curve_keys"


def _candidate_keys(curve) -> list:
    return [k for k in (curve.duplicate_candidates or []) if isinstance(k, str)]


def _curve_data_source(curve):
    owner = curve.material if curve.material_id else curve.sample.material
    return owner.data_source


def prefill_curve_links(context, rows) -> None:
    """Resolve the curve links the rows will publish, in batches.

    Covers the cross-sample duplicate groups behind `curve_assigned_to_several_samples` issues
    (of issue rows, or of property rows with prefetched issues) and the candidate replots of
    curve rows.
    """
    groups = context.setdefault(_CURVE_GROUPS, {})
    keys = context.setdefault(_CURVE_KEYS, {})
    curve_ids, wanted = set(), set()
    for row in rows:
        if isinstance(row, PropertyQualityIssue):
            if row.code in public_projection.CURVE_ARRAY_CODES:
                curve_ids.add(row.property.curve_id)
        elif isinstance(row, Property):
            issues = (getattr(row, "_prefetched_objects_cache", None) or {}).get("quality_issues")
            if issues is not None and any(i.code in public_projection.CURVE_ARRAY_CODES for i in issues):
                curve_ids.add(row.curve_id)
        elif isinstance(row, Curve):
            names = _candidate_keys(row)
            if names:
                source = _curve_data_source(row)
                wanted.update((source, k) for k in names)
    curve_ids = {c for c in curve_ids if c is not None and c not in groups}
    if curve_ids:
        root_of = {pk: dup or pk for pk, dup in
                   Curve.objects.filter(pk__in=curve_ids).values_list("pk", "duplicate_of_id")}
        members = defaultdict(list)
        roots = set(root_of.values())
        for pk, dup in (Curve.objects.filter(Q(pk__in=roots) | Q(duplicate_of_id__in=roots))
                        .values_list("pk", "duplicate_of_id")):
            members[dup or pk].append(pk)
        for curve_id in curve_ids:
            groups[curve_id] = sorted(members.get(root_of.get(curve_id, curve_id), []))
    wanted = {w for w in wanted if w not in keys}
    if wanted:
        sources = {s for s, _ in wanted}
        found = (Curve.objects.filter(source_key__in={k for _, k in wanted})
                 .filter(Q(sample__material__data_source__in=sources) | Q(material__data_source__in=sources))
                 .values_list("pk", "source_key", "sample__material__data_source", "material__data_source"))
        for pk, key, sample_source, material_source in found:
            keys[(sample_source or material_source, key)] = pk
        for w in wanted:
            keys.setdefault(w, None)


class _CurveLinkListSerializer(serializers.ListSerializer):
    """A list whose curve links are resolved for the whole window first (`prefill_curve_links`)."""

    def to_representation(self, data):
        rows = list(data.all() if isinstance(data, dj_models.manager.BaseManager) else data)
        prefill_curve_links(self.context, rows)
        return [self.child.to_representation(item) for item in rows]


def _curve_group_ids(curve_id, context) -> list:
    """Ids of a curve and of the byte-identical curves filed under other samples.

    One `duplicate_of` group in id order, from the window's batch when it was resolved there.
    """
    if curve_id is None:
        return []
    groups = context.get(_CURVE_GROUPS) or {}
    if curve_id not in groups:
        root = Curve.objects.filter(pk=curve_id).values_list("duplicate_of_id", flat=True).first() or curve_id
        groups = context.setdefault(_CURVE_GROUPS, {})
        groups[curve_id] = list(Curve.objects.filter(Q(pk=root) | Q(duplicate_of_id=root)).order_by("pk")
                                .values_list("pk", flat=True))
    return groups[curve_id]


# Quality issues


class PropertyQualityIssueSerializer(_ReadOnly):
    """An objection recorded beside a value, never instead of it.

    Publishes code, severity, the projected observed value, bound and unit.
    """
    observed_value = serializers.SerializerMethodField()

    def get_observed_value(self, obj):
        curve_ids = None
        if obj.code in public_projection.CURVE_ARRAY_CODES:
            curve_ids = _curve_group_ids(obj.property.curve_id, self.context)
        return public_projection.observed_value(obj.code, obj.observed_value, curve_ids=curve_ids)

    class Meta:
        model = PropertyQualityIssue
        fields = ("id", "property", "code", "severity", "observed_value", "bound", "unit", "upstream_reported")
        read_only_fields = fields
        list_serializer_class = _CurveLinkListSerializer


class DeviceQualityIssueSerializer(_ReadOnly):
    """A device's objection, published beside its metrics like a property's (it has no unit)."""
    observed_value = serializers.SerializerMethodField()

    def get_observed_value(self, obj):
        return public_projection.observed_value(obj.code, obj.observed_value)

    class Meta:
        model = DeviceQualityIssue
        fields = ("id", "device", "code", "severity", "metric", "observed_value", "bound", "upstream_reported")
        read_only_fields = fields


class SampleQualityIssueSerializer(_ReadOnly):
    """A sample's objection (no metric, bound or unit; its observed value is text)."""
    observed_value = serializers.SerializerMethodField()

    def get_observed_value(self, obj):
        return public_projection.sample_observed_value(obj.observed_value)

    class Meta:
        model = SampleQualityIssue
        fields = ("id", "sample", "code", "severity", "observed_value", "upstream_reported")
        read_only_fields = fields


# Structures
class CrystalSiteSerializer(_ReadOnly):
    class Meta:
        model = CrystalSite
        fields = ("site", "species_label", "element", "multiplicity", "occupancy", "vacancy", "coordination")
        read_only_fields = fields


class StructureSerializer(_ReadOnly):
    crystal_sites = CrystalSiteSerializer(many=True, read_only=True)
    #: Who determined the symmetry, in words: the stored vocabulary term is not shown.
    symmetry_source = serializers.SerializerMethodField()

    def get_symmetry_source(self, obj) -> Optional[str]:
        return public_projection.symmetry_source_label(obj.symmetry_source)

    class Meta:
        model = Structure
        fields = ("id", "material", "sample", "layer", "data_source", "role", "symmetry_source", "symprec",
                  "structure_type", "space_group", "crystal_system", "a", "b", "c", "alpha", "beta", "gamma",
                  "volume", "coordinates_json", "cif_file", "xyz_file", "mol_file", "pdb_file", "crystal_sites")
        read_only_fields = fields


# Processes, properties, samples
class ProcessSerializer(_ReadOnly):
    """One step (and part) of a sample's preparation; scalar columns hold this step's values."""
    conditions = ProjectedJSONField("Process.conditions")

    class Meta:
        model = Process
        fields = ("id", "material", "sample", "device", "data_source", "step_index", "step_count", "layer_index",
                  "method", "method_raw", "solvents", "aggregation_state", "quenching_media", "quenching_additives",
                  "atmosphere", "temperature_program", "conditions", "precursor_solution", "solvent",
                  "spin_speed_rpm", "spin_time_s", "annealing_temp_c", "annealing_time_min")
        read_only_fields = fields


class PropertySerializer(_ReadOnly):
    quality_issues = PropertyQualityIssueSerializer(many=True, read_only=True)
    context = ProjectedJSONField("Property.context")
    #: Display label for `name` (`vocabulary.DISPLAY_LABELS`); the stored name stays the identifier.
    display_name = serializers.SerializerMethodField()

    def get_display_name(self, obj) -> str:
        from materials.services.ingest.vocabulary import display_label
        return display_label(obj.name)

    class Meta:
        model = Property
        fields = ("id", "material", "sample", "device", "structure", "curve", "data_source", "name",
                  "display_name", "category", "value", "unit", "raw_value", "null_reason", "temperature_k",
                  "method", "reference", "context", "quality_issues")
        read_only_fields = fields
        list_serializer_class = _CurveLinkListSerializer


class SampleSerializer(_ReadOnly):
    """One physical sample, cell, emitter batch or molecule instance as the source describes it."""
    quality_issues = SampleQualityIssueSerializer(many=True, read_only=True)
    conditions = ProjectedJSONField("Sample.conditions")
    bibliography = ProjectedJSONField("Sample.bibliography")

    class Meta:
        model = Sample
        fields = ("id", "material", "label", "form", "data_type", "cited_work", "reference", "figure",
                  "conditions", "bibliography", "quality_issues")
        read_only_fields = fields


# Devices
class DeviceLayerSerializer(_ReadOnly):
    conditions = ProjectedJSONField("DeviceLayer.conditions")

    class Meta:
        model = DeviceLayer
        fields = ("id", "device", "position", "role", "material_text", "thickness_nm", "additives", "deposition",
                  "conditions")
        read_only_fields = fields


class DeviceJVScanSerializer(_ReadOnly):
    """The sweeps behind a device's single quadruple.

    `default_basis` says which sweep each stored device metric is, `quality_flags` what is wrong
    with the numbers.
    """

    class Meta:
        model = DeviceJVScan
        fields = ("id", "device", "direction", "pce", "voc", "jsc", "ff", "hysteresis_index", "quality_flags",
                  "default_basis")
        read_only_fields = fields


DEVICE_FIELDS = ("id", "material", "sample", "data_source", "device_type", "pce", "voc", "jsc", "ff",
                 "ff_raw_value", "ff_raw_unit", "architecture", "test_condition", "reference", "cells_averaged",
                 "area_cm2", "active_area_basis", "polarity", "light_intensity_mw_cm2", "spectrum", "pce_basis",
                 "certified", "encapsulated", "preconditioning", "module", "illumination_class")


class DeviceSerializer(_ReadOnly):
    layers = DeviceLayerSerializer(many=True, read_only=True)
    jv_scans = DeviceJVScanSerializer(many=True, read_only=True)
    quality_issues = DeviceQualityIssueSerializer(many=True, read_only=True)

    class Meta:
        model = Device
        fields = DEVICE_FIELDS + ("layers", "jv_scans", "quality_issues")
        read_only_fields = fields


# Curves
class CurveSerializer(_ReadOnly):
    """A digitized or computed series: axes, status, links, array lengths and a bounded preview.

    The raw arrays (`raw_x`/`raw_y`) are left out. `duplicate_of` is a proven duplicate (a curve
    id); `duplicate_candidates` are possible replots (curve ids), shown apart from it.
    """
    x_context = ProjectedJSONField("Curve.x_context")
    duplicate_candidates = serializers.SerializerMethodField()
    x_points = serializers.SerializerMethodField()
    y_points = serializers.SerializerMethodField()
    preview = serializers.SerializerMethodField()

    def get_duplicate_candidates(self, obj) -> list:
        names = _candidate_keys(obj)
        if not names:
            return []
        source = _curve_data_source(obj)
        if any((source, k) not in (self.context.get(_CURVE_KEYS) or {}) for k in names):
            prefill_curve_links(self.context, [obj])
        keys = self.context[_CURVE_KEYS]
        return sorted({keys[(source, k)] for k in names if keys.get((source, k)) is not None})

    def get_x_points(self, obj) -> int:
        return len(obj.raw_x or [])

    def get_y_points(self, obj) -> int:
        return len(obj.raw_y or [])

    def get_preview(self, obj) -> dict:
        xs, ys = obj.raw_x or [], obj.raw_y or []
        return {"x": list(xs[:CURVE_PREVIEW_POINTS]) if xs else None,
                "y": list(ys[:CURVE_PREVIEW_POINTS]),
                "truncated": len(xs) > CURVE_PREVIEW_POINTS or len(ys) > CURVE_PREVIEW_POINTS}

    class Meta:
        model = Curve
        fields = ("id", "material", "sample", "x_quantity", "x_unit", "y_quantity", "y_unit", "axis_form",
                  "figure", "status", "status_reason", "x_context", "duplicate_of", "duplicate_candidates",
                  "x_points", "y_points", "preview")
        read_only_fields = fields
        list_serializer_class = _CurveLinkListSerializer


# Material serializer
#: Most sub-resources inlined into a single Material payload. The corpus is skewed (one material
#: holds tens of thousands of devices), so inline lists are capped; the complete set stays in the
#: `devices` relation and its siblings.
MAX_INLINE_SUBRESOURCES: int = 100

#: `to_attr` names for the two bounded device reads `prefetch._with_material_prefetch` attaches.
#: The inline window is id-ordered and the champion is PCE-ordered, so the two are fetched separately.
#:
#: Absent on any path that does not prefetch (e.g. a MaterialSerializer built over plain
#: instances), so every reader falls back to a bounded query.
INLINE_DEVICES_ATTR = "_inline_devices"
# Re-exported from `device_policy`, which defines the policy this pool serves, so both modules
# use one attribute name.
CHAMPION_POOL_ATTR = _CHAMPION_POOL_ATTR

#: `to_attr` names for the bounded property and process windows (one material can carry tens of
#: thousands of properties and processes).
INLINE_PROPERTIES_ATTR = "_inline_properties"
INLINE_PROCESSES_ATTR = "_inline_processes"
#: `to_attr` name for the bounded structure window.
INLINE_STRUCTURES_ATTR = "_inline_structures"
#: The material's samples, bounded the same way (a PerovskiteDB family record holds thousands).
INLINE_SAMPLES_ATTR = "_inline_samples"

#: `to_attr` prefix for the one row, chosen in SQL, that answers each published scalar; see
#: `_representative_prop`.
SCALAR_ROW_ATTR = "_scalar_row_"


#: Where `_champion_cell` memoises its answer on the Material instance.
#: pce/voc/jsc/ff each ask for the champion, so an uncached fallback would issue
#: the unbounded query four times for the same material.
_CHAMPION_CACHE_ATTR = "_champion_cell_cache"

#: Where a windowed champion is memoised. Kept apart from `_CHAMPION_CACHE_ATTR`, which is
#: `device_policy._CACHE_ATTR` and shared so that a champion resolved here is not resolved again
#: by `champion_cell()`; that attribute holds a Device or None, never a per-window dict.
_CHAMPION_WINDOW_CACHE_ATTR = "_champion_cell_window_cache"


class _CappedListSerializer(serializers.ListSerializer):
    """Serialize at most `MAX_INLINE_SUBRESOURCES` items from a related manager.

    The slice is applied to the queryset before serialization, so the truncated
    rows are never built. Ordering is by id, so the retained window is stable
    across calls rather than depending on the prefetch cache order (neither
    Property nor Device declares Meta.ordering).
    """

    #: `to_attr` this list serializer should read instead of the manager, when
    #: the caller has already fetched a bounded window under that name.
    bounded_attr: str | None = None

    #: Relations the child serializer nests, loaded for the whole window in one
    #: query when the caller did not prefetch them (a no-op when it did).
    nested_prefetch: tuple = ()

    def _rows(self, rows):
        rows = list(rows)
        if self.nested_prefetch and rows:
            prefetch_related_objects(rows, *self.nested_prefetch)
        prefill_curve_links(self.context, rows)
        return [self.child.to_representation(item) for item in rows]

    def to_representation(self, data):
        # A bounded prefetch under `to_attr` is a plain list on the parent, not the manager the
        # field points at. Read the list when it is there; the manager path below serves callers
        # that build this serializer over un-prefetched instances.
        if self.bounded_attr:
            parent = getattr(data, "instance", None)
            bounded = getattr(parent, self.bounded_attr, None)
            if bounded is not None:
                return self._rows(bounded)
        iterable = data.all() if hasattr(data, "all") else data
        # Slice in Python when the rows are already loaded: `.order_by(...)` on a prefetched
        # manager builds a new queryset, which discards the prefetch cache and issues one query
        # per parent (N+1). Query the database only when the rows were not prefetched.
        cached = getattr(iterable, "_result_cache", None)
        if cached is None and hasattr(iterable, "order_by"):
            iterable = iterable.order_by("id")[:MAX_INLINE_SUBRESOURCES]
        else:
            rows = cached if cached is not None else list(iterable)
            iterable = sorted(rows, key=lambda x: x.id)[:MAX_INLINE_SUBRESOURCES]
        return self._rows(iterable)


# Inline-only subclasses: the cap applies to nested use only, so `DeviceSerializer` and
# `PropertySerializer` used on their own serialize every row they are given.
class _CappedProcessListSerializer(_CappedListSerializer):
    bounded_attr = INLINE_PROCESSES_ATTR


class _InlineProcessSerializer(ProcessSerializer):
    class Meta(ProcessSerializer.Meta):
        list_serializer_class = _CappedProcessListSerializer


class _CappedPropertyListSerializer(_CappedListSerializer):
    bounded_attr = INLINE_PROPERTIES_ATTR


class _InlinePropertySerializer(PropertySerializer):
    class Meta(PropertySerializer.Meta):
        list_serializer_class = _CappedPropertyListSerializer


class _CappedStructureListSerializer(_CappedListSerializer):
    bounded_attr = INLINE_STRUCTURES_ATTR


class _InlineStructureSerializer(StructureSerializer):
    class Meta(StructureSerializer.Meta):
        list_serializer_class = _CappedStructureListSerializer


class _CappedDeviceListSerializer(_CappedListSerializer):
    bounded_attr = INLINE_DEVICES_ATTR
    nested_prefetch = ("quality_issues",)


class _InlineDeviceSerializer(DeviceSerializer):
    """The material payload's device preview, without layers and J-V sweeps.

    Both would cost one query per inlined device; `DeviceSerializer` includes them.
    """

    layers = None        # a declared field is dropped by naming it None
    jv_scans = None

    class Meta(DeviceSerializer.Meta):
        list_serializer_class = _CappedDeviceListSerializer
        fields = DEVICE_FIELDS + ("quality_issues",)
        read_only_fields = fields


class _CappedSampleListSerializer(_CappedListSerializer):
    bounded_attr = INLINE_SAMPLES_ATTR
    nested_prefetch = ("quality_issues",)


class _InlineSampleSerializer(SampleSerializer):
    """The material payload's sample rows with their quality issues.

    The stated conditions and bibliography are left out (a PerovskiteDB sample states up to a few
    hundred conditions); `SampleSerializer` includes them.
    """

    conditions = None
    bibliography = None

    class Meta(SampleSerializer.Meta):
        list_serializer_class = _CappedSampleListSerializer
        fields = ("id", "material", "label", "form", "data_type", "cited_work", "reference", "figure",
                  "quality_issues")
        read_only_fields = fields


def _material_curves(material):
    """The curves of a material: computed curves it owns and the measured curves of its samples."""
    return Curve.objects.filter(Q(material=material) | Q(sample__material=material))



class MaterialSerializer(_ReadOnly):
    # Every nested relation is capped at `MAX_INLINE_SUBRESOURCES` rows.
    structures = _InlineStructureSerializer(many=True, read_only=True)
    processes = _InlineProcessSerializer(many=True, read_only=True)
    properties = _InlinePropertySerializer(many=True, read_only=True)
    devices = _InlineDeviceSerializer(many=True, read_only=True)
    samples = _InlineSampleSerializer(many=True, read_only=True)
    #: The curves of the material and of its samples, with bounded previews; null unless the
    #: context's `detail_of` is "Material".
    curves = serializers.SerializerMethodField()
    identifiers = ProjectedJSONField("Material.identifiers")
    source_attributes = ProjectedJSONField("Material.source_attributes")
    sample_count = serializers.SerializerMethodField()
    curve_count = serializers.SerializerMethodField()

    def get_sample_count(self, obj) -> int:
        annotated = getattr(obj, "_sample_count", None)
        return annotated if annotated is not None else obj.samples.count()

    def get_curve_count(self, obj) -> int:
        annotated = getattr(obj, "_curve_count", None)
        return annotated if annotated is not None else _material_curves(obj).count()

    def get_curves(self, obj):
        if self.context.get(DETAIL_OF) != "Material":
            return None
        rows = (_material_curves(obj).select_related("material", "sample__material")
                .order_by("pk")[:MAX_INLINE_SUBRESOURCES])
        return CurveSerializer(rows, many=True, context=self.context).data
    # Counts so a reader can tell a capped list from a complete one.
    device_count = serializers.SerializerMethodField()
    property_count = serializers.SerializerMethodField()
    process_count = serializers.SerializerMethodField()

    def get_device_count(self, obj) -> int:
        # Prefer the prefetch's annotation; `.count()` covers un-annotated paths.
        annotated = getattr(obj, "_device_count", None)
        if annotated is not None:
            return annotated
        # A full prefetch of the relation is counted in memory. Bounded
        # `to_attr` windows are stored under their own names and never land here.
        cache = getattr(obj, "_prefetched_objects_cache", {})
        if "devices" in cache:
            return len(cache["devices"])
        return obj.devices.count()

    def get_property_count(self, obj) -> int:
        annotated = getattr(obj, "_property_count", None)
        if annotated is not None:
            return annotated
        # Many materials exceed the inline cap: PerovskiteDB records group
        # compositions, so measurements from thousands of papers share one record.
        cache = getattr(obj, "_prefetched_objects_cache", {})
        if "properties" in cache:
            return len(cache["properties"])
        return obj.properties.count()

    def get_process_count(self, obj) -> int:
        """Return the total number of processes; the inline list is capped.

        The complete set is the `processes` relation.
        """
        annotated = getattr(obj, "_process_count", None)
        if annotated is not None:
            return annotated
        cache = getattr(obj, "_prefetched_objects_cache", {})
        if "processes" in cache:
            return len(cache["processes"])
        return obj.processes.count()

    # Computed scalars
    #
    # Two kinds, distinguished by whether a selection policy exists.
    #
    # `pce`/`voc`/`jsc`/`ff` come from `_champion_cell`: the highest-PCE cell that passes an
    # explicit plausibility screen, provenance-scoped, with the device id and reference published
    # beside it. That is a stated curation rule.
    #
    # `bandgap`/`carrier_lifetime`/`mobility`/`stability` have no such rule. Each is the smallest
    # imported measurement, a lower envelope over a record set rather than a property of the
    # material (MAPbI3's band gaps span 1.30-2.90 eV across methods, temperatures, phases and
    # sources). A median is no better: with one source repeated thousands of times the median is
    # that source's, not the material's.
    #
    # Each of these scalars is therefore a lower bound, not a representative value.
    # `property_stats` describes the record set, the provenance-aware property lookup reports
    # every value, and a representative value would need an explicit selection policy (source,
    # method, structure, calculation level, experimental conditions).
    bandgap = serializers.SerializerMethodField()
    pce = serializers.SerializerMethodField()
    # What the published cell rests on: "one_sun" (100 mW/cm², complete and
    # consistent J-V), or, for a material with no such cell, the basis of the
    # best plausible one ("one_sun_incomplete_jv", "other_illumination",
    # "illumination_not_recorded"), or "implausible_record" when no cell of the
    # material passes the plausibility screen. Null when no cell is published.
    pce_basis = serializers.SerializerMethodField()
    voc = serializers.SerializerMethodField()
    jsc = serializers.SerializerMethodField()
    ff = serializers.SerializerMethodField()
    carrier_lifetime = serializers.SerializerMethodField()
    mobility = serializers.SerializerMethodField()
    stability = serializers.SerializerMethodField()
    property_stats = serializers.SerializerMethodField()

    def get_property_stats(self, obj):
        """Per-quantity description of the record set.

        `{"bandgap": {"min": 1.3, "max": 2.9, "count": 26050, "unit": "eV"}}`.

        Computed only where the `_stat_*` aggregates are annotated on the instance, and null elsewhere
        rather than absent, so a reader can tell "not computed" from "the corpus holds none". This
        package does not annotate them, so the `materials` records of `ask` carry null here and the
        values are in the lower-bound scalars and the `properties` relation.

        No median: an aggregate over a source-imbalanced record set describes the records rather
        than the material, so only min/max/count are published.
        """
        stats = {}
        for key in self.RANGE_FILTERED_SCALARS:
            count = getattr(obj, f"_stat_count_{key}", None)
            # Stored rows, suspect ones included; `count`/`min`/`max` describe the eligible
            # values only.
            stored = getattr(obj, f"_stat_stored_count_{key}", None)
            # Skip an absent annotation and an empty record set alike: an all-null
            # entry says nothing a missing key does not.
            if not count and not stored:
                continue
            stats[key] = {
                "min": getattr(obj, f"_stat_min_{key}", None),
                "max": getattr(obj, f"_stat_max_{key}", None),
                "count": count or 0,
                "stored_count": stored if stored is not None else count,
                "unit": getattr(obj, f"_stat_unit_{key}", None),
            }
        return stats or None

    class Meta:
        model = Material
        fields = (
            "id", "name", "formula", "display_formula", "element_formula", "reduced_formula", "composition_status",
            "composition_group", "material_family", "parent_compound", "smiles", "is_lead_free", "description",
            "chemical_class", "structure_type", "crystal_system", "dimensionality", "morphology", "function",
            "device", "data_source", "external_id", "identifiers", "source_attributes",
            "structures", "processes", "properties", "devices", "samples", "curves",
            "device_count", "property_count", "process_count", "sample_count", "curve_count",
            "bandgap", "pce", "pce_basis", "voc", "jsc", "ff", "carrier_lifetime", "mobility", "stability",
            "property_stats",
        )
        read_only_fields = fields

    # Helpers that use the prefetch_related cache (no extra database queries)

    # The four scalars that also take a range filter (`property_bounds`). A material can carry
    # several measurements of one quantity, so the filter, the published value and the ordering
    # share one definition: the smallest imported measurement that satisfies the active range
    # filter, or the smallest of all when no range is active. The value is then inside the range
    # by construction, is the expression the ordering subquery ranks on, and matches what the
    # RAG recommend tool returns.
    RANGE_FILTERED_SCALARS = {
        "bandgap":          {"name": "bandgap"},
        "carrier_lifetime": {"name_contains": "lifetime"},
        "mobility":         {"name_contains": "mobility"},
        "stability":        {"name": "energy_above_hull"},
    }

    def _active_bounds(self, key):
        """(min, max) requested for `key` in the serializer context, or (None, None)."""
        return (self.context.get("property_bounds") or {}).get(key, (None, None))

    def _representative_prop(self, obj, key):
        """Smallest imported measurement of `key` that satisfies the filter.

        `_with_material_prefetch` hands over a one-row window per scalar, chosen in SQL by the same
        ordering (value, then id, so ties are stable: Property has no Meta.ordering) with the same
        provenance and range filters. The Python path serves callers that bypass that prefetch
        (e.g. a serializer built over plain instances).

        "Smallest" is a lower envelope, not a representative value.
        """
        window = getattr(obj, SCALAR_ROW_ATTR + key, None)
        if window is not None:
            return window[0].value if window else None

        selector = self.RANGE_FILTERED_SCALARS[key]
        name = selector.get("name")
        name_contains = selector.get("name_contains")
        low, high = self._active_bounds(key)
        # Eligibility before selection, as in the prefetch's SQL window: a value with a suspect
        # objection is never the published scalar. One query for the material's suspect ids
        # rather than one per row; `is_property_eligible` when the issues came prefetched.
        from materials.services.quality import (
            is_property_eligible, suspect_properties,
        )
        props = list(obj.properties.all())
        if all("quality_issues" in (getattr(p, "_prefetched_objects_cache", None) or {})
               for p in props):
            suspect = {p.pk for p in props if not is_property_eligible(p)}
        else:
            suspect = set(suspect_properties(Property.objects.filter(material=obj))
                          .values_list("pk", flat=True))
        best = None
        for p in props:
            if p.created_by_id is not None:      # imported rows only
                continue
            if p.value is None or p.pk in suspect:
                continue
            if name and p.name.lower() != name.lower():
                continue
            if name_contains and name_contains.lower() not in p.name.lower():
                continue
            if low is not None and p.value < low:
                continue
            if high is not None and p.value > high:
                continue
            if best is None or (p.value, p.id) < (best.value, best.id):
                best = p
        return best.value if best is not None else None

    def _active_device_bounds(self, key):
        """(min, max) requested for a DEVICE field in the serializer context."""
        return (self.context.get("device_bounds") or {}).get(key, (None, None))

    def _champion_cell(self, obj):
        """Return the single solar cell whose PCE, Voc, Jsc and FF are published.

        All four values come from one cell so the published tuple is internally
        consistent (PCE = Voc·Jsc·FF) and attributable to one device. The
        champion is the highest-PCE imported cell; cells with a `created_by` user
        are excluded.

        Uses the PCE-ordered pool `_with_material_prefetch` attaches, otherwise a
        bounded query. The result is memoized on the instance, per PCE window.
        """
        low, high = self._active_device_bounds("pce")
        if low is None and high is None:
            # Unwindowed: the memo shared with `champion_cell()`.
            cached = getattr(obj, _CHAMPION_CACHE_ATTR, "unset")
            if cached != "unset":
                return cached
            champion = self._resolve_champion(obj)
            setattr(obj, _CHAMPION_CACHE_ATTR, champion)
            return champion
        # Windowed: a separate memo keyed by the window, since filtered and
        # unfiltered serializations of one instance differ.
        window = getattr(obj, _CHAMPION_WINDOW_CACHE_ATTR, None)
        if window is None:
            window = {}
            setattr(obj, _CHAMPION_WINDOW_CACHE_ATTR, window)
        key = (low, high)
        if key not in window:
            window[key] = self._resolve_champion(obj, low=low, high=high)
        return window[key]

    @staticmethod
    def _in_window(device, low, high):
        pce = getattr(device, "pce", None)
        if pce is None:
            return low is None and high is None
        return (low is None or pce >= low) and (high is None or pce <= high)

    @classmethod
    def _resolve_champion(cls, obj, *, low=None, high=None):
        """Resolve the champion from the prefetched pool, else via `device_policy`.

        `low`/`high` are the context's PCE filter. Within that window the rule
        is the same (highest-PCE plausible cell), so a filtered serialization
        publishes the champion of the filtered set.
        """
        windowed = low is not None or high is not None
        pool = getattr(obj, CHAMPION_POOL_ATTR, None)
        if pool is None:
            if windowed:
                # No prefetch: query the window.
                candidates = [d for d in imported_solar_cells(obj)
                              if cls._in_window(d, low, high)]
                return pick_champion(candidates)
            return champion_cell(obj)
        if windowed:
            inside = [d for d in pool if cls._in_window(d, low, high)]
            # The pool is the top CHAMPION_POOL_SIZE cells by PCE. If no
            # in-window cell in it is plausible, a plausible one may lie below
            # the pool, so query the full window instead.
            if any(is_plausible_cell(d) for d in inside) or len(pool) < CHAMPION_POOL_SIZE:
                # A short pool is complete, so the fallback applies to it directly.
                champion = pick_champion(inside)
            else:
                champion = pick_champion([d for d in imported_solar_cells(obj)
                                          if cls._in_window(d, low, high)])
            if (champion is not None and not is_one_sun_complete_cell(champion)
                    and len(pool) >= CHAMPION_POOL_SIZE):
                # A one-sun cell inside the window may sit below the bounded
                # pool, and it outranks every cell in the pool.
                window = {}
                if low is not None:
                    window["pce__gte"] = low
                if high is not None:
                    window["pce__lte"] = high
                deeper = imported_solar_cells(obj).filter(one_sun_cell_q(), **window).first()
                if deeper is not None:
                    return deeper
            return champion
        # Unwindowed: the policy's own bounded resolution, which reads this same
        # prefetched pool and looks below it for a one-sun cell when needed.
        return champion_cell(obj)


    # Device and property numbers can legitimately be 0.0 (e.g. PCE=0 for a failed cell), so the
    # getters test `is not None` rather than truthiness.

    def get_bandgap(self, obj) -> Optional[float]:
        """Return the lowest eligible imported band gap: a lower bound, not a representative value.

        Use `property_stats["bandgap"]` or the agent's `lookup_property`, which
        reports every measurement with its source and method.
        """
        return self._representative_prop(obj, 'bandgap')

    # pce/voc/jsc/ff all read the same cell; see `_champion_cell`. Scoping to
    # solar cells keeps LED drive/turn-on voltages out of the photovoltaic `voc`.
    def get_pce(self, obj) -> Optional[float]:
        device = self._champion_cell(obj)
        return device.pce if device is not None else None

    def get_pce_basis(self, obj) -> Optional[str]:
        return pce_basis(self._champion_cell(obj))

    def get_voc(self, obj) -> Optional[float]:
        device = self._champion_cell(obj)
        return device.voc if device is not None else None

    def get_jsc(self, obj) -> Optional[float]:
        device = self._champion_cell(obj)
        return device.jsc if device is not None else None

    def get_ff(self, obj) -> Optional[float]:
        device = self._champion_cell(obj)
        return device.ff if device is not None else None

    def get_carrier_lifetime(self, obj) -> Optional[float]:
        return self._representative_prop(obj, 'carrier_lifetime')

    def get_mobility(self, obj) -> Optional[float]:
        return self._representative_prop(obj, 'mobility')

    def get_stability(self, obj) -> Optional[float]:
        # energy_above_hull (eV/atom) only: category='stability' mixes
        # incompatible quantities (formation energy, total energy, T80 hours).
        return self._representative_prop(obj, 'stability')
