"""Public projection of the corpus JSON columns and quality-issue values.

The corpus stores what each source states, together with derivation metadata that is not public:
rule ids in a few Property.context keys, a classification term in Sample.conditions, the match
basis of equivalence candidates and the raw arrays a quality issue was raised on. The serializers,
Genie and the embedding text read those columns only through this module, so the hidden part never
reaches a reader. Nothing here writes back; the stored rows keep every key.

Each governed JSON column is published through an explicit list of top-level keys. A key the list
does not name is not published. Under a public key the stored value is published as stated (the
open maps of source columns are public), except the nested keys named in `nested_hidden`; every
nested path whose value is a rule id is one of them. A `_*` family of keys is a public prefix. Two
arrays are large and published only with `detail` set (`serializers.DETAIL_OF`): JARVIS-SC
`source_attributes.cfid` and C2DB `context.series`. The `materials` records of `ask` leave them out.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnSpec:
    public: frozenset
    hidden: frozenset = frozenset()
    #: public prefixes (`_*` families of source columns)
    public_prefixes: tuple = ()
    #: public only with `detail` set (large arrays)
    detail_only: frozenset = frozenset()
    #: {public top-level key: nested keys not published}
    nested_hidden: dict = field(default_factory=dict)

    def is_public(self, key: str) -> bool:
        if key in self.hidden:
            return False
        return key in self.public or key in self.detail_only or key.startswith(self.public_prefixes)


def _keys(text: str) -> frozenset:
    return frozenset(text.split())


COLUMNS = {
    # Hidden: the three keys whose values are rule ids. `derived.rule` names a rule by definition.
    "Property.context": ColumnSpec(
        public=_keys("""
            at_luminance atmosphere atmosphere_oxygen_concentration bias burn_in candidate_reading candidates
            cell_atoms cells_averaged cie_basis correction_factor criterion definition deposit_stability derived
            doping_pct duration_h encapsulated excitation_nm field_t graded header_as_stated host hubbard_u_ev
            humidity humidity_pct initial_pce k1_raw_unit k1_temperature_raw_unit k1_temperature_raw_value
            layer_index light light_bias method_basis metric mode orientation outdoor periodic_jv phase pressure_gpa
            protocol qualifier reference_level run_type scan selection_gap sensitizer site solvent source_column
            temperature_c temperature_k temperature_label tensor transition upper value_index x"""),
        hidden=_keys("k1_rule k1_temperature_rule_basis correction_rule"),
        detail_only=_keys("series"),
        nested_hidden={"derived": frozenset({"rule"})},
    ),
    # Every source-stated value; the family disposition is derivation metadata (the statements it
    # classifies stay public).
    "Sample.conditions": ColumnSpec(
        public=_keys("""
            abc3 absorber_unaligned add_lay_back add_lay_front after_treatment after_treatment_detail
            annealing_condition anode_active_material anode_additive_1 anode_areal_capacity anode_binder_1
            anode_conductive_agent anode_current_collector application_as_stated architecture
            band_gap_estimation_basis band_gap_graded boolean_statements candidate_references capping_2d_layer
            cathode_active_material cathode_additive_1 cathode_additive_2 cathode_areal_capacity
            cathode_areal_density_cathode_loading cathode_binder_1 cathode_binder_2 cathode_conductive_agent
            cathode_current_collector cathode_volumetric_density cell_architecture cell_area_total_cm2 cell_type
            cells_per_substrate certification_institute complement composition_basis composition_conversion
            composition_long_form composition_short_form composition_type construction counter_electrode
            device_less dimension_as_stated electrical_measurement electrolyte_additive_1 electrolyte_additive_2
            encapsulation eqe_light_bias eqe_measured excess_components field flexible grain_size inorganic_stated
            jv_illumination_direction jv_light_source_brand jv_light_source_type jv_light_spectra_raw
            jv_light_wavelength_range jv_measured jv_simulator_class jv_storage_age jv_storage_atmosphere
            jv_storage_relative_humidity jv_test_atmosphere jv_test_relative_humidity jv_test_temperature
            lattice_parameter layer_dimensions layer_unaligned lead_free_stated lower_voltage_limit
            magnetic_measurement mask_area_cm2 masked_cell material_family_statements maximum_power
            measurement_condition measurement_temperature mn15si26_001 n null_reasons orientation
            other_sample_process_info peak_efficiency perovskite_inspired preconditioning pressure purity
            quasicrystal_comment quasicrystal_type quenching_flag_conflict quenching_stated
            ratio_of_electrolyte_to_cathode_active_material reference_electrode relative_density
            saturation_magnetization scan_rate semitransparent separator sheet sintering_temperature solute_1
            solute_2 solute_3 solvent_1 solvent_2 solvent_3 solvent_annealing_stated stability_context
            stability_measured stated_step_count stoichiometric synthesis thermal_measurement upper_voltage_limit
            working_electrode"""),
        hidden=_keys("material_family_disposition"),
        public_prefixes=("absorber_", "band_gap_", "jv_scan_", "module_", "stabilised_", "substrate_"),
    ),
    "Sample.bibliography": ColumnSpec(public=_keys("authors journal publication_date")),
    # Stated deposition, storage and treatment conditions, with their null reasons.
    "DeviceLayer.conditions": ColumnSpec(public=_keys("""
            additives_concentrations_raw additives_stated_none after_treatment atmosphere
            deposition_aggregation_state_of_reactants deposition_raw deposition_reaction_solutions_age
            deposition_reaction_solutions_compounds deposition_reaction_solutions_compounds_purity
            deposition_reaction_solutions_compounds_supplier deposition_reaction_solutions_concentrations
            deposition_reaction_solutions_temperature deposition_reaction_solutions_volumes deposition_solvents
            deposition_solvents_mixing_ratios deposition_solvents_purity deposition_solvents_supplier
            deposition_substrate_temperature deposition_synthesis_atmosphere
            deposition_synthesis_atmosphere_pressure_partial deposition_synthesis_atmosphere_pressure_total
            deposition_synthesis_atmosphere_relative_humidity deposition_thermal_annealing_atmosphere
            deposition_thermal_annealing_temperature deposition_thermal_annealing_time null_reasons solvents
            solvents_stated_none storage_atmosphere storage_relative_humidity storage_time_until_next_deposition_step
            surface_treatment_before_next_deposition_step temperature_c thickness_raw time_min""")),
    "Process.conditions": ColumnSpec(public=_keys("""
            annealing_atmosphere annealing_pressure annealing_raw annealing_relative_humidity non_deposition
            precursors pressure_partial pressure_total quenching_additives_stated_none quenching_volume
            relative_humidity solution_age solution_temperature solution_volume solvent_annealing
            solvents_stated_none substrate_temperature""")),
    # `equivalence_candidates` keep the linked record (source, id); their match basis holds
    # derivation metadata.
    "Material.identifiers": ColumnSpec(
        public=_keys("""equivalence_candidates icsd_ids inchi inchikey_measured inchikey_model jarvis_id mp_ids
                        parent_mp_id qmof_refcode smiles_from_inchi smiles_parseable smiles_source source_url"""),
        nested_hidden={"equivalence_candidates": frozenset({"basis"})},
    ),
    "Material.source_attributes": ColumnSpec(
        public=_keys("""calculation cell_formula creation_task_label dimension_as_stated dimensionality_label is_metal
                        layer_number_n literature_doi magnetic_ordering material_family nat origin potcar_titel
                        relative_id sheet source_collection source_id spacegroup structure_drawing structure_origin
                        synthesized task_provenance topology topology_null_reason topology_raw_value"""),
        detail_only=_keys("cfid"),
        # the label and its thresholds are public; `rule` holds the rule id
        nested_hidden={"magnetic_ordering": frozenset({"rule"})},
    ),
    "Curve.x_context": ColumnSpec(public=_keys("quantity scale_k unit")),
}


def _without(value, hidden: frozenset):
    """Return `value` without the `hidden` members of its object(s), applied to each list item."""
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if k not in hidden}
    if isinstance(value, list):
        return [_without(v, hidden) for v in value]
    return value


def project(column: str, value, *, detail: bool = True) -> dict:
    """Return the published part of one stored JSON column value.

    `detail=False` leaves out the `detail_only` keys.
    """
    spec = COLUMNS[column]
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, item in value.items():
        if not spec.is_public(key) or (key in spec.detail_only and not detail):
            continue
        out[key] = _without(item, spec.nested_hidden[key]) if key in spec.nested_hidden else item
    return out


# Structure.symmetry_source
#: The stored vocabulary names who determined the symmetry; the reader is told it in words.
SYMMETRY_SOURCE_LABELS = {
    "source-reported": "reported by the source",
    "LAMP-derived": "derived from the deposited coordinates",
}


def symmetry_source_label(stored):
    return SYMMETRY_SOURCE_LABELS.get(stored) if stored else None


# Quality issue observed_value
#: Codes whose stored observed value is the raw arrays of a curve (StarryData2:
#: `json.dumps({"x": raw_x, "y": raw_y})` of the curve filed under several samples). Published
#: as the linked curves and their point counts; the arrays themselves are read from the curve's
#: `raw_x`/`raw_y`.
CURVE_ARRAY_CODES = frozenset({"curve_assigned_to_several_samples"})


def _parsed(text):
    """Return the JSON value `text` holds, or the text itself when it is not JSON.

    The text is parsed, never prefix-matched.
    """
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def _arrays_summary(parsed) -> dict:
    return {"x_points": len(parsed.get("x") or []), "y_points": len(parsed.get("y") or [])}


def _published(parsed, original):
    """Publish a parsed container: a number list as numbers, a curve object by its point counts."""
    if isinstance(parsed, dict) and isinstance(parsed.get("x", []), list) and isinstance(parsed.get("y", []), list) \
            and ("x" in parsed or "y" in parsed):
        return _arrays_summary(parsed)
    if isinstance(parsed, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in parsed):
        return parsed
    if isinstance(parsed, (dict, list)):
        return parsed
    return original


def observed_value(code: str, stored, *, curve_ids=None):
    """Return the published observed value of a Property- or DeviceQualityIssue (a JSON column).

    Numbers, null and short stated text are published as stored. A string holding JSON is published
    as the value it holds, never as JSON text; the curve arrays of a cross-sample duplicate become
    `{"curves": [ids], "x_points": n, "y_points": n}` (`curve_ids`: the ids of the curves the
    issue links).
    """
    parsed = _parsed(stored)
    if code in CURVE_ARRAY_CODES and isinstance(parsed, dict):
        return {"curves": list(curve_ids or []), **_arrays_summary(parsed)}
    if parsed is stored or not isinstance(parsed, (dict, list)):
        return stored
    return _published(parsed, stored)


def sample_observed_value(stored: str):
    """Return the published observed value of a SampleQualityIssue (a text column).

    Text is published as stored. Only text that parses as a JSON container or null is published as
    the value it holds; a bracket-atom SMILES starts with '[' and is not JSON.
    """
    parsed = _parsed(stored)
    if parsed is None:
        return None
    if isinstance(parsed, (dict, list)):
        return _published(parsed, stored)
    return stored
