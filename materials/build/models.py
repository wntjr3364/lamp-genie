"""Corpus schema: one model per table.

Rules the database can hold are constraints here (unique source keys, one curve owner, a NULL value
with its reason) and constraint triggers in migration 0002 (the ownership rules).

Text columns are TextField so that source notation is kept whole. A column whose NULL is a stated
state (a NULL value with a reason, an optional number) is nullable; an absent text is blank; an
absent JSON value is an empty container.
"""
from django.conf import settings
from django.db import models
from django.db.models.functions import Now

from materials.build import vocab
from materials.build.vocab import choices


class CorpusModel(models.Model):
    """Timestamps are volatile: set by the database so COPY needs no value for them."""
    created_at = models.DateTimeField(db_default=Now())
    updated_at = models.DateTimeField(db_default=Now())

    class Meta:
        abstract = True


def _fk(to, null=False, related_name="+", **kw):
    """A corpus foreign key; `related_name` is the reverse name the code reads (no column)."""
    return models.ForeignKey(to, on_delete=models.CASCADE, null=null, blank=null, related_name=related_name, **kw)


def _user():
    return models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
                             related_name="+")


def _run():
    return models.ForeignKey("corpus.BuildRun", on_delete=models.PROTECT, related_name="+")


class BuildRun(CorpusModel):
    """The import that loaded the tables."""
    target_db = models.TextField()
    git_head = models.TextField()
    inputs_digest = models.TextField()
    environment_digest = models.TextField()
    contract_digest = models.TextField()
    rule_versions = models.JSONField(default=dict)
    id_reservations = models.JSONField(default=dict)
    started_at = models.DateTimeField(db_default=Now())
    finished_at = models.DateTimeField(null=True, blank=True)
    counts = models.JSONField(default=dict)
    status = models.CharField(max_length=10, choices=choices(vocab.BUILD_RUN_STATUSES))

    class Meta:
        db_table = "corpus_build_run"


class Material(CorpusModel):
    data_source = models.CharField(max_length=30, choices=choices(vocab.DATA_SOURCES))
    external_id = models.TextField()
    name = models.TextField()
    formula = models.TextField(blank=True)
    display_formula = models.TextField(blank=True)
    element_formula = models.TextField(blank=True)
    reduced_formula = models.TextField(blank=True)
    smiles = models.TextField(blank=True)
    composition_status = models.CharField(max_length=20, choices=choices(vocab.COMPOSITION_STATUS))
    composition_group = models.TextField(blank=True)
    material_family = models.TextField(blank=True)
    parent_compound = models.TextField(blank=True)
    chemical_class = models.CharField(max_length=30, blank=True, choices=choices(vocab.CHEMICAL_CLASSES))
    structure_type = models.CharField(max_length=30, blank=True, choices=choices(vocab.STRUCTURE_TYPES))
    crystal_system = models.CharField(max_length=20, blank=True, choices=choices(vocab.MATERIAL_CRYSTAL_SYSTEMS))
    dimensionality = models.CharField(max_length=2, blank=True, choices=choices(vocab.DIMENSIONALITIES))
    morphology = models.CharField(max_length=30, blank=True, choices=choices(vocab.MORPHOLOGIES))
    function = models.CharField(max_length=30, blank=True, choices=choices(vocab.FUNCTIONS))
    device = models.CharField(max_length=20, blank=True, choices=choices(vocab.DEVICES))
    is_lead_free = models.BooleanField(null=True, blank=True)
    description = models.TextField(blank=True)
    classification_basis = models.JSONField(default=dict, blank=True)
    identifiers = models.JSONField(default=dict, blank=True)
    source_attributes = models.JSONField(default=dict, blank=True)
    created_by = _user()

    class Meta:
        db_table = "corpus_material"
        constraints = [
            models.UniqueConstraint(fields=["data_source", "external_id"], name="corpus_material_key"),
            models.CheckConstraint(condition=~models.Q(external_id=""), name="corpus_material_external_id"),
        ]


class Sample(CorpusModel):
    """One physical sample, cell, emitter batch or molecule instance as the source describes it."""
    material = _fk(Material, related_name='samples')
    source_key = models.TextField()
    label = models.TextField(blank=True)
    form = models.TextField(null=True, blank=True)
    data_type = models.CharField(max_length=12, null=True, blank=True, choices=choices(vocab.SAMPLE_DATA_TYPES))
    cited_work = models.TextField(null=True, blank=True)
    reference = models.TextField(null=True, blank=True)
    figure = models.TextField(blank=True)
    note = models.TextField(blank=True)
    conditions = models.JSONField(default=dict, blank=True)
    bibliography = models.JSONField(default=dict, blank=True)
    build_run = _run()

    class Meta:
        db_table = "corpus_sample"
        indexes = [models.Index(fields=["source_key"], name="corpus_sample_source_key")]


class Device(CorpusModel):
    sample = _fk(Sample, related_name='devices')
    material = _fk(Material, related_name='devices')
    data_source = models.CharField(max_length=30, choices=choices(vocab.DATA_SOURCES))
    source_key = models.TextField()
    pce = models.FloatField(null=True, blank=True)
    voc = models.FloatField(null=True, blank=True)
    jsc = models.FloatField(null=True, blank=True)
    ff = models.FloatField(null=True, blank=True)
    ff_raw_value = models.TextField(blank=True)
    ff_raw_unit = models.TextField(blank=True)
    ff_rule_basis = models.TextField(blank=True)
    device_type = models.TextField(blank=True)
    architecture = models.TextField(blank=True)
    test_condition = models.TextField(blank=True)
    reference = models.TextField(blank=True)
    note = models.TextField(blank=True)
    cells_averaged = models.IntegerField(null=True, blank=True)
    area_cm2 = models.FloatField(null=True, blank=True)
    active_area_basis = models.CharField(max_length=10, blank=True, choices=choices(vocab.ACTIVE_AREA_BASES))
    polarity = models.CharField(max_length=3, null=True, blank=True, choices=choices(vocab.POLARITIES))
    light_intensity_mw_cm2 = models.FloatField(null=True, blank=True)
    spectrum = models.TextField(null=True, blank=True)
    pce_basis = models.CharField(max_length=20, blank=True, choices=choices(vocab.PCE_BASES))
    certified = models.BooleanField(null=True, blank=True)
    encapsulated = models.BooleanField(null=True, blank=True)
    preconditioning = models.TextField(blank=True)
    module = models.BooleanField(null=True, blank=True)
    illumination_class = models.CharField(max_length=10, null=True, blank=True, choices=choices(vocab.ILLUMINATION_CLASSES))
    created_by = _user()
    build_run = _run()

    class Meta:
        db_table = "corpus_device"
        constraints = [models.UniqueConstraint(fields=["data_source", "source_key"], name="corpus_device_key")]


class DeviceLayer(CorpusModel):
    device = _fk(Device, related_name='layers')
    source_key = models.TextField()
    position = models.IntegerField(null=True, blank=True)
    role = models.CharField(max_length=15, blank=True, choices=choices(vocab.LAYER_ROLES))
    material_text = models.TextField(blank=True)
    thickness_nm = models.FloatField(null=True, blank=True)
    additives = models.JSONField(default=list, blank=True)
    deposition = models.CharField(max_length=100, blank=True, choices=choices(vocab.LAYER_DEPOSITIONS))
    note = models.TextField(blank=True)
    conditions = models.JSONField(default=dict, blank=True)
    build_run = _run()

    class Meta:
        db_table = "corpus_device_layer"
        constraints = [models.UniqueConstraint(fields=["device", "source_key"], name="corpus_device_layer_key")]


class DeviceJVScan(CorpusModel):
    device = _fk(Device, related_name='jv_scans')
    data_source = models.CharField(max_length=30, choices=choices(vocab.DATA_SOURCES))
    source_key = models.TextField()
    direction = models.CharField(max_length=12, choices=choices(vocab.JV_DIRECTIONS))
    pce = models.FloatField(null=True, blank=True)
    voc = models.FloatField(null=True, blank=True)
    jsc = models.FloatField(null=True, blank=True)
    ff = models.FloatField(null=True, blank=True)
    hysteresis_index = models.FloatField(null=True, blank=True)
    quality_flags = models.TextField(blank=True)
    default_basis = models.JSONField(null=True, blank=True)
    quality_context = models.JSONField(default=dict, blank=True)
    note = models.TextField(blank=True)
    build_run = _run()

    class Meta:
        db_table = "corpus_device_jv_scan"
        constraints = [models.UniqueConstraint(fields=["data_source", "source_key"], name="corpus_device_jv_scan_key")]


class Structure(CorpusModel):
    material = _fk(Material, related_name='structures')
    sample = _fk(Sample, null=True, related_name='structures')
    layer = _fk(DeviceLayer, null=True, related_name='structures')
    data_source = models.CharField(max_length=30, choices=choices(vocab.DATA_SOURCES))
    source_key = models.TextField()
    role = models.CharField(max_length=30, choices=choices(vocab.STRUCTURE_ROLES))
    symmetry_source = models.CharField(max_length=20, null=True, blank=True, choices=choices(vocab.SYMMETRY_SOURCES))
    symprec = models.FloatField(null=True, blank=True)
    ingestion = models.JSONField(default=dict, blank=True)
    structure_type = models.TextField(blank=True)
    space_group = models.TextField(blank=True)
    crystal_system = models.CharField(max_length=20, blank=True, choices=choices(vocab.STRUCTURE_CRYSTAL_SYSTEMS))
    a = models.FloatField(null=True, blank=True)
    b = models.FloatField(null=True, blank=True)
    c = models.FloatField(null=True, blank=True)
    alpha = models.FloatField(null=True, blank=True)
    beta = models.FloatField(null=True, blank=True)
    gamma = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    note = models.TextField(blank=True)
    coordinates_json = models.JSONField(null=True, blank=True)
    cif_file = models.FileField(upload_to="structures/", blank=True)
    xyz_file = models.FileField(upload_to="structures/", blank=True)
    mol_file = models.FileField(upload_to="structures/", blank=True)
    pdb_file = models.FileField(upload_to="structures/", blank=True)
    created_by = _user()
    build_run = _run()

    class Meta:
        db_table = "corpus_structure"
        constraints = [models.UniqueConstraint(fields=["data_source", "source_key"], name="corpus_structure_key")]


class CrystalSite(CorpusModel):
    structure = _fk(Structure, related_name='crystal_sites')
    site = models.TextField()
    species_label = models.TextField(blank=True)
    element = models.TextField(blank=True)
    multiplicity = models.FloatField(null=True, blank=True)
    occupancy = models.FloatField(null=True, blank=True)
    vacancy = models.BooleanField(default=False)
    coordination = models.IntegerField(null=True, blank=True)
    note = models.TextField(blank=True)

    class Meta:
        db_table = "corpus_crystal_site"
        constraints = [models.UniqueConstraint(fields=["structure", "site", "species_label", "element"],
                                               name="corpus_crystal_site_key")]


class Curve(CorpusModel):
    """A digitized or computed series owned by a Sample (measured) or a Material (computed)."""
    sample = _fk(Sample, null=True, related_name='curves')
    material = _fk(Material, null=True, related_name='curves')
    source_key = models.TextField()
    x_quantity = models.TextField(blank=True)
    x_unit = models.TextField(blank=True)
    y_quantity = models.TextField(blank=True)
    y_unit = models.TextField(blank=True)
    axis_form = models.CharField(max_length=10, choices=choices(vocab.CURVE_AXIS_FORMS))
    figure = models.TextField(blank=True)
    status = models.CharField(max_length=10, choices=choices(vocab.CURVE_STATUSES))
    status_reason = models.TextField(blank=True)
    duplicate_of = _fk("self", null=True, related_name='curve_set')
    duplicate_candidates = models.JSONField(default=list, blank=True)
    note = models.TextField(blank=True)
    x_context = models.JSONField(default=dict, blank=True)
    raw_x = models.JSONField(default=list, blank=True)
    raw_y = models.JSONField(default=list, blank=True)
    build_run = _run()

    class Meta:
        db_table = "corpus_curve"
        indexes = [models.Index(fields=["source_key"], name="corpus_curve_source_key")]
        constraints = [models.CheckConstraint(condition=models.Q(sample__isnull=True) ^ models.Q(material__isnull=True),
                                              name="corpus_curve_one_owner")]


class Process(CorpusModel):
    """One step (and part) of a sample's preparation; scalar columns hold this step's values."""
    material = _fk(Material, related_name='processes')
    sample = _fk(Sample, related_name='processes')
    device = _fk(Device, null=True, related_name='processes')
    data_source = models.CharField(max_length=30, choices=choices(vocab.DATA_SOURCES))
    source_key = models.TextField()
    step_index = models.IntegerField(null=True, blank=True)
    step_count = models.IntegerField(null=True, blank=True)
    layer_index = models.IntegerField(null=True, blank=True)
    method = models.CharField(max_length=100, blank=True, choices=choices(vocab.PROCESS_METHODS))
    method_raw = models.TextField(blank=True)
    solvents = models.JSONField(default=list, blank=True)
    aggregation_state = models.TextField(blank=True)
    quenching_media = models.JSONField(default=list, blank=True)
    quenching_additives = models.JSONField(default=list, blank=True)
    atmosphere = models.TextField(blank=True)
    temperature_program = models.JSONField(default=list, blank=True)
    conditions = models.JSONField(default=dict, blank=True)
    precursor_solution = models.TextField(blank=True)
    solvent = models.TextField(blank=True)
    spin_speed_rpm = models.IntegerField(null=True, blank=True)
    spin_time_s = models.IntegerField(null=True, blank=True)
    annealing_temp_c = models.FloatField(null=True, blank=True)
    annealing_time_min = models.FloatField(null=True, blank=True)
    note = models.TextField(blank=True)
    created_by = _user()
    build_run = _run()

    class Meta:
        db_table = "corpus_process"
        constraints = [
            models.UniqueConstraint(fields=["data_source", "source_key"], name="corpus_process_key"),
            models.CheckConstraint(condition=models.Q(step_index__isnull=True) | models.Q(step_index__gte=1),
                                   name="corpus_process_step_index"),
        ]


class Property(CorpusModel):
    material = _fk(Material, related_name='properties')
    sample = _fk(Sample, null=True, related_name='properties')
    device = _fk(Device, null=True, related_name='properties')
    structure = _fk(Structure, null=True, related_name='properties')
    curve = _fk(Curve, null=True, related_name='properties')
    data_source = models.CharField(max_length=30, choices=choices(vocab.DATA_SOURCES))
    source_key = models.TextField(blank=True)
    name = models.TextField()
    category = models.TextField(blank=True)
    value = models.FloatField(null=True, blank=True)
    unit = models.TextField(null=True, blank=True)
    raw_value = models.TextField(blank=True)
    null_reason = models.TextField(blank=True)
    temperature_k = models.FloatField(null=True, blank=True)
    method = models.TextField(blank=True)
    reference = models.TextField(blank=True)
    note = models.TextField(blank=True)
    context = models.JSONField(default=dict, blank=True)
    created_by = _user()
    build_run = _run()

    class Meta:
        db_table = "corpus_property"
        indexes = [models.Index(fields=["data_source", "source_key"], name="corpus_property_source_key"),
                   models.Index(fields=["name"], name="corpus_property_name")]
        constraints = [
            # every NULL value carries its reason, and a value carries none
            models.CheckConstraint(condition=models.Q(value__isnull=True, null_reason__gt="")
                                   | models.Q(value__isnull=False, null_reason=""), name="corpus_property_null_reason"),
        ]


# Quality issues
class PropertyQualityIssue(CorpusModel):
    property = _fk(Property, related_name='quality_issues')
    code = models.CharField(max_length=60, choices=choices(vocab.PQI_CODES))
    severity = models.CharField(max_length=10, choices=choices(vocab.SEVERITIES))
    rule = models.TextField(blank=True)
    rule_id = models.TextField()
    rule_version = models.IntegerField()
    observed_value = models.JSONField(null=True, blank=True)
    bound = models.FloatField(null=True, blank=True)
    unit = models.TextField(blank=True)
    assumptions = models.TextField(blank=True)
    upstream_reported = models.BooleanField(default=False)
    note = models.TextField(blank=True)
    detected_by_run = _run()

    class Meta:
        db_table = "corpus_property_quality_issue"
        constraints = [models.UniqueConstraint(fields=["property", "rule_id"], name="corpus_pqi_key")]


class DeviceQualityIssue(CorpusModel):
    device = _fk(Device, related_name='quality_issues')
    code = models.CharField(max_length=60, choices=choices(vocab.DQI_CODES))
    severity = models.CharField(max_length=10, choices=choices(vocab.SEVERITIES))
    metric = models.CharField(max_length=3, blank=True, choices=choices(vocab.DQI_METRICS))
    rule = models.TextField(blank=True)
    rule_id = models.TextField()
    rule_version = models.IntegerField()
    observed_value = models.JSONField(null=True, blank=True)
    bound = models.FloatField(null=True, blank=True)
    assumptions = models.TextField(blank=True)
    upstream_reported = models.BooleanField(default=False)
    note = models.TextField(blank=True)
    detected_by_run = _run()

    class Meta:
        db_table = "corpus_device_quality_issue"
        constraints = [models.UniqueConstraint(fields=["device", "rule_id"], name="corpus_dqi_key")]


class SampleQualityIssue(CorpusModel):
    sample = _fk(Sample, related_name='quality_issues')
    code = models.CharField(max_length=60, choices=choices(vocab.SQI_CODES))
    severity = models.CharField(max_length=10, choices=choices(vocab.SEVERITIES))
    rule_id = models.TextField()
    locator = models.TextField(blank=True)
    observed_value = models.TextField(blank=True)
    assumptions = models.TextField(blank=True)
    upstream_reported = models.BooleanField(default=False)
    note = models.TextField(blank=True)
    detected_by_run = _run()

    class Meta:
        db_table = "corpus_sample_quality_issue"
        constraints = [models.UniqueConstraint(fields=["sample", "locator", "rule_id"], name="corpus_sqi_key")]
