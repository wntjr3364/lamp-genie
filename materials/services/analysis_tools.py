"""Physics-based analysis tools the Genie agent can call."""
from typing import Dict, Any, List, Optional
import math
import re

# Coefficient parsing shared by every formula parser in the package (formula_utils).
from .formula_utils import coefficient as _coefficient
from .formula_utils import FormulaExpansionLimitError, expand_groups, iter_tokens


class BandgapPredictor:
    """Bandgap prediction with bowing parameters"""
    
    BASE_BANDGAPS = {
        "MAPbI3": 1.55,
        "FAPbI3": 1.48,
        "CsPbI3": 1.73,
        "MAPbBr3": 2.25,
        "FAPbBr3": 2.23,
        "CsPbBr3": 2.30,
        "MAPbCl3": 3.11,
        "FAPbCl3": 3.05,
        "CsPbCl3": 3.06,
        "MASnI3": 1.20,
        "FASnI3": 1.41,
        "CsSnI3": 1.27,
        # Sn-Br endpoints (literature values).
        "MASnBr3": 2.15,
        "FASnBr3": 2.40,  # Jokar 2018 reports ~2.40 eV
        "CsSnBr3": 1.75,
        # Sn-Cl: measured gaps of the bulk compounds (photoelectron spectroscopy and optical
        # absorption), Tao et al., Nat. Commun. 10, 2560 (2019): MASnCl3 3.50, FASnCl3 3.55,
        # CsSnCl3 2.88 eV.
        "MASnCl3": 3.50,
        "FASnCl3": 3.55,
        "CsSnCl3": 2.88,
        # Ge B-site (third common lead-free perovskite cation). Iodide gaps are
        # experimentally well-characterised (Krishnamoorthy et al., J. Mater.
        # Chem. A 3, 23829 (2015); Stoumpos et al., J. Am. Chem. Soc. 137, 6804
        # (2015)). Br/Cl-Ge gaps have sparser experimental data; the MA/FA-Br/Cl
        # values are extrapolated from the CsGeX3 series and the iodide A-site shift.
        "MAGeI3": 2.00,
        "FAGeI3": 2.35,
        "CsGeI3": 1.63,
        "MAGeBr3": 2.60,
        "FAGeBr3": 2.80,
        "CsGeBr3": 2.38,
        "MAGeCl3": 3.70,
        "FAGeCl3": 3.90,
        "CsGeCl3": 3.40,
    }
    
    BOWING_PARAMETERS = {
        "I-Br": 0.33,
        "Br-Cl": 0.27,
        "I-Cl": 0.45,
        # Pb-Sn iodide perovskites show a large optical bowing: mixed Pb-Sn
        # alloys dip below both endpoint gaps (anomalous, the basis of
        # narrow-gap tandem bottom cells). The experimental literature reports
        # b ≈ 0.6-0.9 eV; 0.65 eV is a representative mid-range value.
        # Refs: Hao 2014 (JACS), Goyal 2018, Rajagopal 2017 (ACS Energy Lett.).
        "Pb-Sn": 0.65,
    }

    # Quantum-confinement shift for 2D Ruddlesden-Popper perovskites.
    # ΔEg(n) ≈ CONFINEMENT_COEFF_2D / n² (n = number of inorganic layers).
    # 0.5 eV matches the experimentally observed 0.5 eV blueshift for
    # n=1 (BA)₂PbI₄ relative to 3D MAPbI₃. See BandgapPredictor.predict().
    CONFINEMENT_COEFF_2D = 0.5  # eV
    # 0D strong-confinement limit used as a flat additive correction.
    CONFINEMENT_SHIFT_0D = 1.0  # eV
    # Halide perovskites have an anomalous (positive) dEg/dT, opposite to
    # conventional III-V semiconductors: MAPbI3 ≈ +3.4e-4 eV/K (Foley 2015),
    # CsPbI3 ≈ +5e-4 eV/K; Sn-based compounds are roughly opposite. The default
    # is the Pb-iodide value, the most common composition.
    VARSHNI_DEG_DT = +0.0003  # eV/K (positive, anomalous for halide perovskites)
    REFERENCE_TEMPERATURE_K = 300  # K

    @classmethod
    def predict(
        cls,
        a_site: Dict[str, float],  # {"FA": 0.9, "MA": 0.1, "Cs": 0}
        b_site: Dict[str, float],  # {"Pb": 1.0, "Sn": 0}
        x_site: Dict[str, float],  # {"I": 0.8, "Br": 0.2, "Cl": 0}
        dimensionality: str = "3D",
        layer_number: int = 1,
        temperature: float = 300,
    ) -> Dict[str, Any]:
        """
        Predict the band gap (eV) with bowing, confinement and temperature corrections.

        Site fractions are normalized per site. Returns a dict with "error" and
        None values for unsupported or empty sites.

        Returns:
            {
                "bandgap": float,
                "linear_bandgap": float,
                "bowing_correction": float,
                "dimension_correction": float,
                "temperature_correction": float,
            }
        """
        a_total = sum(a_site.values())
        b_total = sum(b_site.values())
        x_total = sum(x_site.values())

        # Reject input with no halide or no cation rather than assume one.
        if x_total <= 0:
            return {
                "error": "X-site (halide) composition sums to zero; cannot predict bandgap without a halide.",
                "bandgap": None,
                "linear_bandgap": None,
                "bowing_correction": None,
                "dimension_correction": None,
                "temperature_correction": None,
            }
        if a_total <= 0 or b_total <= 0:
            return {
                "error": "Missing A-site or B-site cation; cannot predict bandgap.",
                "bandgap": None,
                "linear_bandgap": None,
                "bowing_correction": None,
                "dimension_correction": None,
                "temperature_correction": None,
            }

        a_site = {k: v/a_total for k, v in a_site.items()}
        b_site = {k: v/b_total for k, v in b_site.items()}
        x_site = {k: v/x_total for k, v in x_site.items()}

        # The interpolation is parametrized for A-site FA/MA/Cs and B-site
        # Pb/Sn/Ge only. 2D-spacer A-cations need a confinement model and Bi
        # double perovskites (A2BB'X6) a different structure, so other cations
        # are rejected.
        _SUPPORTED_A = {"FA", "MA", "Cs"}
        _SUPPORTED_B = {"Pb", "Sn", "Ge"}
        unsupported = sorted(
            [k for k, v in a_site.items() if v > 0 and k not in _SUPPORTED_A]
            + [k for k, v in b_site.items() if v > 0 and k not in _SUPPORTED_B]
        )
        if unsupported:
            return {
                "error": (
                    "Bandgap prediction is parametrized only for A-site FA/MA/Cs and "
                    f"B-site Pb/Sn/Ge. Unsupported cation(s): {', '.join(unsupported)} "
                    "(e.g. Bi/Ag double perovskites and 2D-spacer A-cations need a "
                    "different model)."
                ),
                "bandgap": None,
                "linear_bandgap": None,
                "bowing_correction": None,
                "dimension_correction": None,
                "temperature_correction": None,
            }

        # Linear interpolation (Vegard's law) over the endpoint table:
        # Eg_linear = Σ_{A,B,X} f_A·f_B·f_X · Eg(ABX3). Adding a cation needs only
        # its BASE_BANDGAPS rows and an entry in the _SUPPORTED sets above, which
        # guarantee every key looked up here is tabulated.
        linear_eg = 0.0
        for b_cat, b_frac in b_site.items():
            if b_frac <= 0:
                continue
            for x_hal in ("I", "Br", "Cl"):
                x_frac = x_site.get(x_hal, 0)
                if x_frac <= 0:
                    continue
                for a_cat, a_frac in a_site.items():
                    if a_frac <= 0:
                        continue
                    linear_eg += (
                        a_frac * b_frac * x_frac
                        * cls.BASE_BANDGAPS[f"{a_cat}{b_cat}{x_hal}3"]
                    )

        bowing_correction = 0
        i = x_site.get("I", 0)
        br = x_site.get("Br", 0)
        cl = x_site.get("Cl", 0)
        
        if i > 0 and br > 0:
            bowing_correction -= cls.BOWING_PARAMETERS["I-Br"] * i * br
        if br > 0 and cl > 0:
            bowing_correction -= cls.BOWING_PARAMETERS["Br-Cl"] * br * cl
        if i > 0 and cl > 0:
            bowing_correction -= cls.BOWING_PARAMETERS["I-Cl"] * i * cl
        
        if b_site.get("Pb", 0) > 0 and b_site.get("Sn", 0) > 0:
            bowing_correction -= cls.BOWING_PARAMETERS["Pb-Sn"] * b_site["Pb"] * b_site["Sn"]
        
        # Dimensionality effect (quantum confinement)
        dimension_correction = 0
        if dimensionality == "2D" and layer_number > 0:
            # ΔEg ≈ CONFINEMENT_COEFF_2D / n² for 2D perovskites
            dimension_correction = cls.CONFINEMENT_COEFF_2D / (layer_number ** 2)
        elif dimensionality == "0D":
            dimension_correction = cls.CONFINEMENT_SHIFT_0D  # strong confinement

        # Temperature correction (linearized Varshni approximation)
        # Eg(T) ≈ Eg(300) + (dEg/dT)*(T - 300)
        temperature_correction = cls.VARSHNI_DEG_DT * (temperature - cls.REFERENCE_TEMPERATURE_K)
        
        bandgap = linear_eg + bowing_correction + dimension_correction + temperature_correction
        
        return {
            "bandgap": round(bandgap, 3),
            "linear_bandgap": round(linear_eg, 3),
            "bowing_correction": round(bowing_correction, 3),
            "dimension_correction": round(dimension_correction, 3),
            "temperature_correction": round(temperature_correction, 3),
        }


class BandAlignmentCalculator:
    """Band offsets and heterojunction type for each interface of a device stack."""

    # Energy levels in eV below vacuum (Schulz 2019, Endres 2016, Brivio 2014);
    # semiconductors satisfy IE - EA = Eg.
    PRESET_MATERIALS = {
        # Absorbers (perovskites)
        "FAPbI3": {"work_function": 4.5, "electron_affinity": 3.90, "ionization_energy": 5.38, "bandgap": 1.48},
        "MAPbI3": {"work_function": 4.6, "electron_affinity": 3.93, "ionization_energy": 5.48, "bandgap": 1.55},
        "CsPbI3": {"work_function": 4.85, "electron_affinity": 3.95, "ionization_energy": 5.68, "bandgap": 1.73},
        # Schulz 2019, Brivio 2014.
        "CsPbBr3": {"work_function": 4.50, "electron_affinity": 3.60, "ionization_energy": 5.90, "bandgap": 2.30},
        # Br -> Cl deepens the VBM.
        "CsPbCl3": {"work_function": 4.00, "electron_affinity": 3.05, "ionization_energy": 6.15, "bandgap": 3.10},
        "MAPbBr3": {"work_function": 4.95, "electron_affinity": 3.38, "ionization_energy": 5.68, "bandgap": 2.30},
        "FAPbBr3": {"work_function": 4.90, "electron_affinity": 3.47, "ionization_energy": 5.70, "bandgap": 2.23},
        "MASnI3": {"work_function": 4.20, "electron_affinity": 4.10, "ionization_energy": 5.40, "bandgap": 1.30},
        "FASnI3": {"work_function": 4.20, "electron_affinity": 4.19, "ionization_energy": 5.60, "bandgap": 1.41},
        "CsSnI3": {"work_function": 4.20, "electron_affinity": 4.13, "ionization_energy": 5.40, "bandgap": 1.27},
        # Electron transport layers (ETL); Schulz 2019, Jiang 2017
        "SnO2": {"work_function": 4.50, "electron_affinity": 4.30, "ionization_energy": 7.90, "bandgap": 3.60},
        "TiO2": {"work_function": 4.20, "electron_affinity": 4.20, "ionization_energy": 7.40, "bandgap": 3.20},
        "ZnO": {"work_function": 4.30, "electron_affinity": 4.30, "ionization_energy": 7.50, "bandgap": 3.20},
        "PCBM": {"work_function": 4.20, "electron_affinity": 3.90, "ionization_energy": 6.10, "bandgap": 2.20},
        "C60": {"work_function": 4.50, "electron_affinity": 4.00, "ionization_energy": 6.40, "bandgap": 2.40},
        # Hole transport layers (HTL)
        "Spiro": {"work_function": 5.00, "electron_affinity": 2.05, "ionization_energy": 5.05, "bandgap": 3.00},
        "PTAA": {"work_function": 5.10, "electron_affinity": 2.30, "ionization_energy": 5.20, "bandgap": 2.90},
        # NiOx EA from UPS (Manders 2013)
        "NiOx": {"work_function": 5.20, "electron_affinity": 1.40, "ionization_energy": 5.00, "bandgap": 3.60},
        "PEDOT:PSS": {"work_function": 5.00, "electron_affinity": 2.10, "ionization_energy": 5.10, "bandgap": 3.00},
        "CuSCN": {"work_function": 5.40, "electron_affinity": 1.80, "ionization_energy": 5.50, "bandgap": 3.70},
        # Electrodes are metals (bandgap=0), handled by the Schottky branch of
        # `calculate()`. Only WF is physical; EA and IE are set equal to it.
        "ITO": {"work_function": 4.70, "electron_affinity": 4.70, "ionization_energy": 4.70, "bandgap": 0},
        "FTO": {"work_function": 4.40, "electron_affinity": 4.40, "ionization_energy": 4.40, "bandgap": 0},
        "Au": {"work_function": 5.10, "electron_affinity": 5.10, "ionization_energy": 5.10, "bandgap": 0},
        "Ag": {"work_function": 4.70, "electron_affinity": 4.70, "ionization_energy": 4.70, "bandgap": 0},
        "Al": {"work_function": 4.20, "electron_affinity": 4.20, "ionization_energy": 4.20, "bandgap": 0},
        "Cu": {"work_function": 4.65, "electron_affinity": 4.65, "ionization_energy": 4.65, "bandgap": 0},
    }

    # Case-sensitive aliases for common name variants
    PRESET_ALIASES = {
        "Spiro-OMeTAD": "Spiro",
        "spiro-OMeTAD": "Spiro",
        "spiro": "Spiro",
        "NiO": "NiOx",
        "NiO_x": "NiOx",
        "Niox": "NiOx",
        "PEDOT": "PEDOT:PSS",
        "PEDOT-PSS": "PEDOT:PSS",
    }
    
    @classmethod
    def calculate(cls, layers: List[str], overrides: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Calculate band offsets for a device stack.

        Args:
            layers: List of material names, e.g., ["ITO", "SnO2", "FAPbI3", "Spiro", "Au"]
            overrides: Stated levels per layer ({name: {"electron_affinity",
                "ionization_energy"}}), which take precedence over the table.

        Returns:
            {
                "layers": [...],
                "offsets": [{"interface": str, "cbo": float, "vbo": float, "type": str}],
                "diagram_data": [...] for visualization
            }
        """
        # Lower-case aliases, tried before PRESET_ALIASES.
        _ALIASES = {
            "spiro-ometad": "Spiro",
            "spiro ometad": "Spiro",
            "spiroometad": "Spiro",
            "sno₂": "SnO2",
            "tio₂": "TiO2",
            "fapbi₃": "FAPbI3",
            "mapbi₃": "MAPbI3",
        }
        layer_data = []
        missing_layers = []
        # Stated levels override the table value by value (see
        # `services.stated_levels`); level_sources reports which were stated.
        stated = {}
        for name, levels in (overrides or {}).items():
            key = _ALIASES.get(name.lower()) or cls.PRESET_ALIASES.get(name, name)
            stated[key] = {k: float(v) for k, v in levels.items()
                           if k in ("electron_affinity", "ionization_energy") and v is not None}
        level_sources = {}
        for layer_name in layers:
            # Lower-case alias, then PRESET_ALIASES, then the name as given.
            canonical = _ALIASES.get(layer_name.lower())
            if canonical is None:
                canonical = cls.PRESET_ALIASES.get(layer_name, layer_name)
            given = stated.get(canonical, {})
            if canonical not in cls.PRESET_MATERIALS and len(given) < 2:
                missing_layers.append(layer_name)
                continue
            row = {"name": canonical, **cls.PRESET_MATERIALS.get(canonical, {}), **given}
            if len(given) == 2:
                # Both levels stated: the gap follows from them, not the table.
                row["bandgap"] = round(row["ionization_energy"] - row["electron_affinity"], 6)
                if row["bandgap"] == 0:
                    # Equal levels describe a metal; that level is its work function.
                    row["work_function"] = row["electron_affinity"]
            if canonical not in cls.PRESET_MATERIALS:
                # Only the stated levels are known: the gap follows from them.
                row["bandgap"] = round(row["ionization_energy"] - row["electron_affinity"], 6)
                row.setdefault("work_function", (row["ionization_energy"] + row["electron_affinity"]) / 2)
            layer_data.append(row)
            level_sources[canonical] = ("stated" if len(given) == 2 else
                                        "partly stated" if given else "table")
        
        # Band offsets (Anderson's electron affinity rule):
        #   E_CB = -EA and E_VB = -IE relative to the vacuum level.
        #   cbo = EA2 - EA1: positive means layer2's CBM is below layer1's.
        #   vbo = IE2 - IE1: positive means layer2's VBM is below layer1's.
        # Heterojunction type depends on the relative signs:
        #   Type-I  (straddling, nested): layer2's gap fits inside layer1's,
        #           i.e. cbo>0 and vbo<0. Inverted (layer1 inside layer2): cbo<0, vbo>0.
        #   Type-II (staggered): cbo and vbo have the same sign.
        #   Type-III (broken): staggered and the gaps do not overlap
        #           (CBM of low-CB side sits below VBM of high-VB side).
        offsets = []
        for i in range(len(layer_data) - 1):
            layer1 = layer_data[i]
            layer2 = layer_data[i + 1]

            cbo = layer2["electron_affinity"] - layer1["electron_affinity"]
            vbo = layer2["ionization_energy"] - layer1["ionization_energy"]

            # Metal contacts (bandgap=0) form Schottky or ohmic contacts, not
            # Type-I/II/III heterojunctions; classify them by work function.
            l1_metal = layer1.get("bandgap", 1.0) == 0
            l2_metal = layer2.get("bandgap", 1.0) == 0
            if l1_metal or l2_metal:
                wf1 = layer1["work_function"]
                wf2 = layer2["work_function"]
                if l1_metal and l2_metal:
                    alignment_type = "Metal-Metal contact"
                else:
                    # Metal/semiconductor contact. Report the smaller of the two
                    # barriers, i.e. the carrier the contact selects:
                    #   φ_Bn = WF(metal) − EA(semi)      (barrier to the CBM)
                    #   φ_Bp = IE(semi) − WF(metal)      (barrier to the VBM)
                    metal, semi = (layer1, layer2) if l1_metal else (layer2, layer1)
                    wf = metal["work_function"]
                    phi_bn = wf - semi["electron_affinity"]
                    phi_bp = semi["ionization_energy"] - wf
                    if phi_bn <= phi_bp:
                        carrier, barrier = "n-type", phi_bn
                    else:
                        carrier, barrier = "p-type", phi_bp
                    alignment_type = (
                        f"Schottky contact ({carrier}, φ_B = {barrier:+.3f} eV)"
                        if barrier > 0.1 else
                        f"Ohmic contact ({carrier}, |φ_B| = {abs(barrier):.3f} eV)"
                    )
            elif cbo > 0 and vbo < 0:
                alignment_type = "Type-I"
            elif cbo < 0 and vbo > 0:
                alignment_type = "Type-I (inverted)"
            else:
                # Same sign: staggered. Check for a broken gap.
                # Energies below vacuum: CB1=-EA1, VB1=-IE1, CB2=-EA2, VB2=-IE2.
                cb1, vb1 = -layer1["electron_affinity"], -layer1["ionization_energy"]
                cb2, vb2 = -layer2["electron_affinity"], -layer2["ionization_energy"]
                # Broken if neither pair overlaps (the lower CB is still below the higher VB).
                if min(cb1, cb2) > max(vb1, vb2):
                    alignment_type = "Type-II"
                else:
                    alignment_type = "Type-III (broken)"

            offsets.append({
                "interface": f"{layer1['name']} / {layer2['name']}",
                "cbo": round(cbo, 3),
                "vbo": round(vbo, 3),
                "type": alignment_type
            })
        
        return {
            "layers": layer_data,
            "offsets": offsets,
            "stack": " / ".join(layers),
            "missing_layers": missing_layers,
            "level_sources": level_sources,
        }


class FormulaValidator:
    """Chemical formula validator and analyzer."""

    # IUPAC element symbols (H..Og, Z=1..118). Element-shaped tokens that are
    # not real elements (e.g. "Xy" in "Xyzzy") are rejected.
    _KNOWN_ELEMENTS = frozenset({
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    })

    # Pseudo-cation tokens recognised as valid "species" alongside real
    # elements (organic A-site cations in hybrid perovskites):
    #   FA formamidinium, MA methylammonium, EA ethylammonium,
    #   BA butylammonium, PA propylammonium.
    _PSEUDO_CATIONS = frozenset({"FA", "MA", "EA", "BA", "PA"})

    @classmethod
    def validate(cls, formula: str) -> Dict[str, Any]:
        """
        Validate and analyze a chemical formula.

        Returns:
            {
                "valid": bool,
                "formula": str,
                "elements": Dict[str, float],
                "charge_balance": bool,
                "charge": charge_assignment(...) -- the states and their sum,
                "is_perovskite": bool,
                "structure": structure_family(...) -- the family from the ratio,
                "warnings": List[str],
                "suggestions": List[str],
            }
        """
        warnings = []
        suggestions = []

        try:
            elements = cls._parse_formula(formula)
        except FormulaExpansionLimitError as exc:
            # A formula past the expansion bounds is refused, not truncated:
            # the input (a tool argument) has no length bound.
            return {
                "valid": False,
                "formula": formula,
                "elements": {},
                "charge_balance": False,
                "is_perovskite": False,
                "warnings": [f"Formula rejected: {exc}"],
                "suggestions": [
                    "Shorten the formula or reduce the number of nested groups."
                ],
            }

        if not elements:
            return {
                "valid": False,
                "formula": formula,
                "elements": {},
                "charge_balance": False,
                "is_perovskite": False,
                "warnings": ["Could not parse formula"],
                "suggestions": ["Check formula syntax"],
            }

        # Reject formulas with characters the tokenizer could not consume
        # (e.g. "Asdfgh" would otherwise parse as {As}).
        residue = cls._unparsed_residue(formula)
        if residue:
            return {
                "valid": False,
                "formula": formula,
                "elements": elements,
                "charge_balance": False,
                "is_perovskite": False,
                "warnings": [
                    f"Could not fully parse formula; unrecognised fragment: '{residue}'"
                ],
                "suggestions": ["Check formula syntax against the periodic table"],
            }

        # Reject element-shaped tokens that are neither real elements nor
        # recognised pseudo-cations (e.g. "Fo").
        unknown_tokens = sorted(
            tok for tok in elements
            if tok not in cls._KNOWN_ELEMENTS and tok not in cls._PSEUDO_CATIONS
        )
        if unknown_tokens:
            return {
                "valid": False,
                "formula": formula,
                "elements": elements,
                "charge_balance": False,
                "is_perovskite": False,
                "warnings": [
                    f"Unrecognised element symbol(s): {', '.join(unknown_tokens)}"
                ],
                "suggestions": [
                    "Check element symbols against the periodic table; "
                    "only real elements and organic cations (FA, MA, EA, BA, PA) are accepted"
                ],
            }

        is_perovskite = cls._check_perovskite(elements)
        
        # Charge balance check, with the assignment that decided it
        charge = cls.charge_assignment(elements)
        charge_balanced = charge["status"] != "unbalanced"
        structure = cls.structure_family(elements, charge)
        
        if "Pb" in elements:
            warnings.append("Contains lead (Pb) - toxic material")
        if "Sn" in elements:
            warnings.append("Contains tin (Sn) - susceptible to oxidation")
        
        if not charge_balanced:
            suggestions.append("Check stoichiometry - charge may not be balanced")
        if is_perovskite and "Pb" in elements:
            suggestions.append("Consider Pb-free alternatives: Sn, Ge, or double perovskites")
        
        return {
            "valid": True,
            "formula": formula,
            "elements": elements,
            "charge_balance": charge_balanced,
            "charge": charge,
            "is_perovskite": is_perovskite,
            "structure": structure,
            "warnings": warnings,
            "suggestions": suggestions,
        }
    
    @classmethod
    def _parse_formula(cls, formula: str) -> Dict[str, float]:
        """Parse chemical formula into elements and counts.

        Organic perovskite cations (FA, MA, ...) are single tokens, and
        parenthesised groups with multipliers (e.g. `Pb(I0.8Br0.2)3`) are
        expanded first so inner counts are scaled.
        """
        expanded = cls._expand_parens(formula)

        # Use the shared scanner: its spacer rule (whether the prefix of an
        # all-caps run is a chain of pseudo-cations) cannot be expressed as a
        # regex alternation.
        matches = [
            (token, count)
            for token, count, _start, _end in iter_tokens(expanded)
            if token[0].isupper()
        ]

        elements: Dict[str, float] = {}
        for element, count in matches:
            elements[element] = elements.get(element, 0) + _coefficient(count)

        return elements

    @classmethod
    def _unparsed_residue(cls, formula: str) -> str:
        """Return the characters left over after stripping valid tokens.

        A well-formed formula consists only of element/cation tokens with
        optional counts, brackets and whitespace; anything else (e.g. the
        "dfgh" in "Asdfgh") is returned. Returns "" for a fully parseable
        formula. A dot is legal only inside a coefficient, i.e. inside a
        matched token span.
        """
        expanded = cls._expand_parens(formula)
        # Same scanner as `_parse_formula`, in strict mode: the coefficient
        # group does not accept a bare ".". Token spans (not substitution) let
        # a dot inside a coefficient be told apart from a stray one, e.g. the
        # ".5" left over from `Pb0.5.5I3`.
        gaps: list[str] = []
        cursor = 0
        for token, _count, start, end in iter_tokens(expanded, strict=True):
            if not token[:1].isupper():
                # Untokenisable characters come back as single raw characters
                # and stay in the gap text; every recognised token starts with
                # a capital letter.
                continue
            gaps.append(expanded[cursor:start])
            cursor = end
        gaps.append(expanded[cursor:])
        residue = "".join(gaps)
        # A dot outside every token span is malformed. Checked before the
        # strip below, which would remove the neighbouring digits. Leftover bare
        # digits are tolerated: hydrate multipliers (`CuSO4·5H2O`) and leading
        # counts (`3PbI3`) are not supported but not flagged either.
        if "." in residue:
            return residue
        # Brackets, digits, whitespace and · or * are structural.
        residue = re.sub(r'[()\[\]\d\s·*]', "", residue)
        # Also scan the raw input for orphan dots, which `_expand_parens` can
        # rewrite away (e.g. "Pb(I.)3" expands to "PbI3").
        residue += "".join(re.findall(r'(?<!\d)\.(?!\d)', formula))
        return residue

    @classmethod
    def _expand_parens(cls, formula: str, max_iter: int = 10) -> str:
        """Expand `(...)N` and `[...]N` groups by multiplying inner counts by N.

        Delegates to `formula_utils.expand_groups`; `max_iter` is the maximum
        nesting depth.
        """
        return expand_groups(formula, max_depth=max_iter)

    # Site vocabularies of the perovskite family. Chalcogenide (S/Se/Te) and
    # pnictide (N) X-sites cover non-halide perovskites (e.g. BaZrS3).
    _A_SITE = frozenset({"FA", "MA", "EA", "BA", "PA", "Cs", "Rb", "K", "Ba", "Sr", "Ca", "Li", "Na"})
    _B_SITE = frozenset({"Pb", "Sn", "Ge", "Ti", "Bi", "Sb", "In", "Ag", "Cu",
                         "Fe", "Mn", "Zr", "Hf", "Nb", "Ta"})
    _X_SITE = frozenset({"I", "Br", "Cl", "O", "F", "S", "Se", "Te", "N"})

    @classmethod
    def _check_perovskite(cls, elements: Dict[str, float]) -> bool:
        """Whether the formula has A-, B- and X-site species of the perovskite family.

        This is element presence only; which family the stoichiometry actually
        is comes from `structure_family`.
        """
        return (any(k in elements for k in cls._A_SITE)
                and any(k in elements for k in cls._B_SITE)
                and any(k in elements for k in cls._X_SITE))

    @classmethod
    def structure_family(cls, elements: Dict[str, float],
                         charge: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Name the perovskite family from the A:B:X stoichiometry.

        Ratios are taken per B-site total:
        - A:B:X = 1:1:3 -> "ABX3"; with exactly two B species in equal amounts at
          different oxidation states (Ag+/Bi3+) -> "A2BB'X6" (elpasolite);
        - 2:1:6 -> "A2BX6" (vacancy-ordered);
        - (1+1/n):1:(3+1/n) for integer n -> "Ruddlesden-Popper n=<n>".
        Anything else, or a species outside the three site vocabularies, gets
        family None with the computed ratio, so a formula with perovskite
        elements is never labelled ABX3 by the elements alone.
        """
        sites = {"A": {}, "B": {}, "X": {}}
        other = []
        for e, n in elements.items():
            for site, vocab in (("A", cls._A_SITE), ("B", cls._B_SITE), ("X", cls._X_SITE)):
                if e in vocab:
                    sites[site][e] = n
                    break
            else:
                other.append(e)
        a, b, x = (sum(sites[k].values()) for k in "ABX")
        out = {"family": None, "sites": sites, "ratio": None, "reason": None}
        if not (a and b and x):
            out["reason"] = "no A-, B- and X-site species together"
            return out
        ra, rx = a / b, x / b
        out["ratio"] = f"A:B:X = {ra:.2f} : 1 : {rx:.2f}"
        if other:
            out["reason"] = "species outside the perovskite sites: " + ", ".join(other)
            return out
        tol = 0.02
        if abs(ra - 1) <= tol and abs(rx - 3) <= tol:
            family = "ABX3"
            b_species = list(sites["B"].items())
            if len(b_species) == 2 and abs(b_species[0][1] - b_species[1][1]) <= tol * b:
                states = {t["species"]: t["state"] for t in (charge or {}).get("terms", [])}
                if not states:
                    states = {e: cls._OXIDATION_STATES.get(e, [None])[0] for e, _ in b_species}
                s0, s1 = (states.get(e) for e, _ in b_species)
                if s0 is not None and s1 is not None and s0 != s1:
                    family = "A2BB'X6"
            out["family"] = family
        elif abs(ra - 2) <= tol and abs(rx - 6) <= tol:
            out["family"] = "A2BX6"
        elif rx > 3 + tol:
            n = 1 / (rx - 3)
            if abs(n - round(n)) <= 0.05 and round(n) >= 1 and abs(ra - (1 + 1 / round(n))) <= tol:
                out["family"] = f"Ruddlesden-Popper n={round(n)}"
        if out["family"] is None:
            out["reason"] = "stoichiometry matches no perovskite ratio"
        return out

    # Allowed oxidation states per element, most common first. A formula is
    # charge-balanced if any combination sums to zero, so e.g. Cs2SnI6 (Sn⁴⁺,
    # vacancy-ordered) passes.
    _OXIDATION_STATES = {
        # Monovalent organic cations
        "FA": [1], "MA": [1], "EA": [1], "BA": [1], "PA": [1],
        # Monovalent inorganic cations
        "Cs": [1], "Rb": [1], "K": [1], "Na": [1], "Li": [1], "H": [1],
        # Multi-valent cations
        "Ag": [1, 2], "Cu": [1, 2],
        "Ba": [2], "Sr": [2], "Ca": [2], "Mg": [2],
        "Pb": [2, 4], "Sn": [2, 4], "Ge": [2, 4],
        "Zn": [2], "Cd": [2], "Ni": [2, 3],
        "Fe": [2, 3], "Mn": [2, 3, 4, 7],
        # Trivalent
        "Bi": [3, 5], "Sb": [3, 5], "In": [3, 1], "La": [3], "Y": [3],
        "Cr": [3, 6], "Al": [3], "Ga": [3],
        # Tetravalent
        "Ti": [4, 3], "Zr": [4], "Hf": [4], "Si": [4], "Ce": [3, 4],
        # Pentavalent
        "V": [5, 4, 3], "Nb": [5], "Ta": [5], "W": [6, 4], "Mo": [6, 4],
        # Anions
        "I": [-1], "Br": [-1], "Cl": [-1], "F": [-1],
        "O": [-2], "S": [-2], "Se": [-2], "Te": [-2],
        "N": [-3], "P": [-3, -2, 3, 5], "C": [-4, 4],
    }

    @classmethod
    def charge_assignment(cls, elements: Dict[str, float]) -> Dict[str, Any]:
        """Return the oxidation states behind the charge-balance verdict, and their sum.

        Every combination of each species' allowed states is tried in order,
        the allowed-state lists being ordered most common first, and the first
        combination that sums to 0 (within 0.1) is reported: species keep their
        usual state for as long as balance allows, in formula order (PbMnO3 is
        Pb2+ Mn4+, not Pb4+ Mn2+). When none does, the usual states are reported
        with their (non-zero) sum, and `allowed` lists what else was tried.

        Returns {"status": "balanced" | "unbalanced" | "not_determined",
        "terms": [{"species", "count", "state", "charge"}], "total",
        "allowed": {species: [states]}, "reason"}.
        """
        from itertools import product
        unknown = [e for e in elements if e not in cls._OXIDATION_STATES]
        if unknown:
            return {"status": "not_determined", "terms": [], "total": None, "allowed": {},
                    "reason": "no oxidation state is known for " + ", ".join(unknown)}
        species = list(elements.items())
        options = [cls._OXIDATION_STATES[e] for e, _ in species]
        allowed = {e: list(cls._OXIDATION_STATES[e]) for e, _ in species}
        # Bound the combinatorial search (the formula is a tool argument); above
        # the cap, report not_determined.
        if math.prod(len(o) for o in options) > 100_000:
            return {"status": "not_determined", "terms": [], "total": None, "allowed": allowed,
                    "reason": "too many oxidation-state combinations to search"}

        def terms_for(states):
            return [{"species": e, "count": n, "state": st, "charge": n * st}
                    for (e, n), st in zip(species, states)]

        best = None
        for states in product(*options):
            if abs(sum(n * st for (_, n), st in zip(species, states))) <= 0.1:
                best = states
                break
        if best is not None:
            terms = terms_for(best)
            return {"status": "balanced", "terms": terms,
                    "total": round(sum(t["charge"] for t in terms), 10),
                    "allowed": allowed, "reason": None}
        terms = terms_for([o[0] for o in options])
        return {"status": "unbalanced", "terms": terms,
                "total": round(sum(t["charge"] for t in terms), 10),
                "allowed": allowed,
                "reason": "no combination of the allowed oxidation states sums to 0"}

    # A-site oxidation states n_A for the Bartel 2019 τ factor:
    # τ = (r_X / r_B) − n_A · (n_A − (r_A/r_B) / ln(r_A/r_B))
    # τ < 4.18: predicted stable perovskite; τ > 4.18: likely unstable
    # (Bartel et al., Sci. Adv. 5, eaav0693, 2019).
    _A_OXIDATION = {
        "FA": 1, "MA": 1, "Cs": 1, "Rb": 1, "K": 1, "Na": 1, "Li": 1,
        "Ba": 2, "Sr": 2, "Ca": 2,
    }

    @classmethod
    def bartel_tau_factor(
        cls,
        a_radius: float,
        b_radius: float,
        x_radius: float,
        n_a: int = 1,
    ) -> Optional[float]:
        """Compute the Bartel 2019 τ stability factor.

        Returns τ, or None when r_A/r_B <= 1 or the inputs are otherwise
        unphysical. τ < 4.18 indicates a likely-stable perovskite (Bartel et al.
        2019: 576 experimental compounds, 92% accuracy vs 74% for t).
        """
        if b_radius <= 0 or a_radius / b_radius <= 1.0:
            return None
        try:
            log_ratio = math.log(a_radius / b_radius)
            if log_ratio == 0:
                return None
            tau = (x_radius / b_radius) - n_a * (n_a - (a_radius / b_radius) / log_ratio)
            return round(tau, 3)
        except (ValueError, ZeroDivisionError):
            return None


#: What the geometric windows do not show, stated with every verdict.
SCREENING_NOTE = ("The tolerance and octahedral windows are an empirical geometric screen and do "
                  "not establish phase stability or perovskite formation.")


def within_screening_window(result) -> bool:
    """Return whether a geometric-stability result lies inside both windows.

    Falls back to the "stability" text for results without the boolean flag.
    """
    flag = (result or {}).get("within_screening_window")
    if isinstance(flag, bool):
        return flag
    return str((result or {}).get("stability") or "").lower().startswith(("stable", "within"))


class StabilityAnalyzer:
    """Stability analysis: tolerance factor, octahedral factor and degradation risks."""

    #: Goldschmidt tolerance-factor window: the decision-tree bounds of Bartel et al., Sci. Adv. 5,
    #: eaav0693 (2019), 0.825 < t < 1.059 on 576 experimental ABX3 compounds.
    T_WINDOW = (0.825, 1.059)
    #: Octahedral-factor window: 0.414 (the octahedral radius-ratio limit) to 0.732
    #: (Pauling's CN=6 limit).
    MU_WINDOW = (0.414, 0.732)

    @staticmethod
    def window_label(window) -> str:
        """A window as stated, three decimals each edge: "0.825–1.059", "0.414–0.732"."""
        return f"{window[0]:.3f}–{window[1]:.3f}"
    
    # Shannon ionic radii (Å): A-site effective molecular/ionic radius,
    # B-site VI-coordinate, X-site VI-coordinate
    # Refs: Shannon 1976, Kieslich et al. Chem. Sci. 2014, Travis et al. Chem. Sci. 2016
    IONIC_RADII = {
        # A-site cations: all on the 12-coordinate scale, as the Goldschmidt
        # tolerance factor requires (the A site sits in the 12-coordinate
        # cuboctahedral cavity). The organic-cation effective radii (Kieslich 2014)
        # are 12-coordinate, so the alkali metals use Shannon's 12-coordinate radii
        # as well; 6-coordinate values would understate Cs/Rb/K and lower every
        # Cs-containing tolerance factor (CsPbI3: 0.851 with these radii, 0.807 with 6-coordinate ones).
        # Refs: Shannon 1976 (XII-coord), Kieslich et al. Chem. Sci. 2014.
        "FA": 2.53,   # Formamidinium effective radius (Kieslich 2014, XII-equiv)
        "MA": 2.17,   # Methylammonium effective radius (Kieslich 2014, XII-equiv)
        "EA": 2.74,   # Ethylammonium  (C2) effective radius (Kieslich 2014 / Travis 2016)
        # Linear alkylammonium effective radii increase monotonically with chain
        # length (each added CH2 enlarges the cation), so propylammonium (C3) is
        # smaller than butylammonium (C4). Values are on the same Kieslich-derived
        # effective-radius scale as EA=2.74 and approximate: C3+ alkylammonium
        # cations mainly template 2D layered perovskites, where the 3D Goldschmidt
        # factor is only indicative. Refs: Kieslich et al. Chem. Sci. 2014;
        # Becker et al. 2017.
        "PA": 2.92,   # Propylammonium (C3)
        "BA": 3.21,   # Butylammonium  (C4)
        "Cs": 1.88,   # Cs+ (XII-coordinate, Shannon 1976)
        "Rb": 1.72,   # Rb+ (XII-coordinate, Shannon 1976)
        "K":  1.64,   # K+  (XII-coordinate, Shannon 1976)
        # B-site cations
        "Pb": 1.19,   # Pb2+ (VI-coordinate, Shannon 1976)
        # Shannon 1976 tabulates no Sn(II) radius at any coordination number.
        # Travis et al., Chem. Sci. 7, 4548 (2016), Table 1 carries 1.15 A in its
        # "Six coordinate Shannon ionic radius" column with footnote c: "Shannon
        # does not give a Sn(II) radius, yet 1.15 A has been used by others."
        # That is the same column this table takes Pb (1.19) and Ge (0.73) from,
        # so 1.15 is the internally consistent value.
        "Sn": 1.15,   # Sn2+ (VI-coordinate; Travis et al. 2016 Table 1, col. 1)
        "Ge": 0.73,   # Ge2+ (VI-coordinate, Shannon 1976)
        "Bi": 1.03,   # Bi3+ (VI-coordinate, Shannon 1976), for Bi-based double perovskites
        # X-site halides (VI-coordinate)
        "I":  2.2,
        "Br": 1.96,
        "Cl": 1.81,
    }
    
    @classmethod
    def analyze_geometric(
        cls,
        a_site: Dict[str, float],
        b_site,  # str "Pb"/"Sn" OR Dict[str, float] for mixed B-site
        x_site: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Calculate tolerance factor and octahedral factor.

        b_site accepts either:
          - A single element string ("Pb" or "Sn") for a pure B site.
          - A Dict[str, float] like {"Pb": 0.6, "Sn": 0.4} for a mixed B site;
            the ionic radius is then the occupancy-weighted average.

        Returns:
            {
                "tolerance_factor": float,
                "octahedral_factor": float,
                "stability": str,               # within or outside the screening window
                "within_screening_window": bool,
                "phase_prediction": str,        # SCREENING_NOTE: what the windows do not show
            }
        """
        # Normalize the A- and X-site fractions.
        a_total = sum(a_site.values())
        x_total = sum(x_site.values())

        if a_total > 0:
            a_site = {k: v/a_total for k, v in a_site.items()}
        if x_total > 0:
            x_site = {k: v/x_total for k, v in x_site.items()}

        # A-site cations with a tabulated effective or ionic radius. A composition
        # with any other A-site species is rejected rather than given a radius of 0,
        # which would yield a physically impossible tolerance factor.
        _KNOWN_A_SITE = ("FA", "MA", "EA", "BA", "PA", "Cs", "Rb", "K")
        unknown_a = [
            ion for ion, frac in a_site.items()
            if frac > 0 and (ion not in _KNOWN_A_SITE or ion not in cls.IONIC_RADII)
        ]
        if unknown_a:
            return {
                "error": (
                    "A-site contains cation(s) with no tabulated radius: "
                    f"{', '.join(sorted(unknown_a))}. Supported A-site species: "
                    f"{', '.join(_KNOWN_A_SITE)}."
                ),
                "tolerance_factor": None,
                "octahedral_factor": None,
                "stability": "Undefined — unsupported A-site cation",
                "phase_prediction": "N/A",
            }
        r_a = sum(
            a_site.get(ion, 0) * cls.IONIC_RADII[ion]
            for ion in _KNOWN_A_SITE
            if ion in cls.IONIC_RADII
        )
        # Resolve b_site to a single effective radius: a dict for a mixed B site,
        # a str for a single element.
        if isinstance(b_site, dict):
            # Reject any B cation with no tabulated radius, as for the A site.
            # Otherwise an unknown element adds its fraction to the normalization
            # denominator but nothing to r_b, understating the effective B radius
            # (Pb0.5Zn0.5 would give mu=0.27). The r_b<=0 check below only catches
            # the all-unknown case, not a partially known mix.
            unknown_b = [
                el for el, frac in b_site.items()
                if frac > 0 and el not in cls.IONIC_RADII
            ]
            if unknown_b:
                return {
                    "error": (
                        "B-site contains cation(s) with no tabulated radius: "
                        f"{', '.join(sorted(unknown_b))}. Supported B-site species are "
                        "those in the ionic-radius table (e.g. Pb, Sn, Ge, Bi)."
                    ),
                    "tolerance_factor": None,
                    "octahedral_factor": None,
                    "stability": "Undefined — unsupported B-site cation",
                    "phase_prediction": "N/A",
                }
            b_total = sum(b_site.values()) or 1.0
            b_normalized = {k: v / b_total for k, v in b_site.items()}
            r_b = sum(
                frac * cls.IONIC_RADII[el]
                for el, frac in b_normalized.items()
                if el in cls.IONIC_RADII
            )
            if r_b <= 0:
                return {
                    "error": "B-site contains no known cation (expected Pb, Sn, or mix).",
                    "tolerance_factor": None,
                    "octahedral_factor": None,
                    "stability": "Undefined — unknown B-site cation",
                    "phase_prediction": "N/A",
                }
        else:
            r_b = cls.IONIC_RADII[b_site]
        r_x = (
            x_site.get("I", 0) * cls.IONIC_RADII["I"] +
            x_site.get("Br", 0) * cls.IONIC_RADII["Br"] +
            x_site.get("Cl", 0) * cls.IONIC_RADII["Cl"]
        )

        # An empty X site (no halide) would divide by zero in the octahedral factor.
        if r_x <= 0:
            return {
                "error": "X-site (halide) composition sums to zero; cannot compute tolerance / octahedral factor.",
                "tolerance_factor": None,
                "octahedral_factor": None,
                "stability": "Undefined — no halide provided",
                "phase_prediction": "N/A",
            }

        # Tolerance factor: t = (r_A + r_X) / [√2 * (r_B + r_X)]
        t = (r_a + r_x) / (math.sqrt(2) * (r_b + r_x))

        # Octahedral factor: μ = r_B / r_X
        mu = r_b / r_x
        
        # The windows are a geometric screen, not a phase prediction: the octahedral window
        # rejects reported perovskites such as the germanium iodides (μ below 0.414).
        (t_lo, t_hi), (mu_lo, mu_hi) = cls.T_WINDOW, cls.MU_WINDOW
        within = t_lo <= t <= t_hi and mu_lo <= mu <= mu_hi

        def place(name, value, lo, hi, lo_text, hi_text):
            # Four decimals: a value just past an edge must not print as the edge itself.
            where = "below" if value < lo else "above" if value > hi else "within"
            return f"{name} = {value:.4f} lies {where} the {lo_text}–{hi_text} window"

        t_edges, mu_edges = cls.window_label(cls.T_WINDOW).split("–"), cls.window_label(cls.MU_WINDOW).split("–")
        factors = (f"{place('t', t, t_lo, t_hi, *t_edges)} and "
                   f"{place('μ', mu, mu_lo, mu_hi, *mu_edges)}")
        stability = (f"{'Within' if within else 'Outside'} the empirical geometric screening "
                     f"window: {factors}")

        return {
            "tolerance_factor": round(t, 3),
            "octahedral_factor": round(mu, 3),
            "stability": stability,
            "within_screening_window": within,
            "phase_prediction": SCREENING_NOTE,
        }
    
    @classmethod
    def analyze_degradation_risks(
        cls,
        composition: str,  # e.g., "FAPbI3"
        dimensionality: str = "3D",
        temperature: float = 25,
        humidity: float = 50,
    ) -> Dict[str, Any]:
        """
        Score heuristic degradation risks (0-1) from composition and conditions.

        Returns:
            {
                "halide_segregation": {"risk": float, "description": str},
                "thermal_degradation": {"risk": float, "description": str},
                "moisture_sensitivity": {"risk": float, "description": str},
                "ion_migration": {"risk": float, "description": str},
                "overall_risk": str,
            }
        """
        risks = {}

        # Halide segregation: any two distinct halides count as mixed. The
        # lookahead `(?![a-z])` keeps "I" from matching inside "In" or "Ir".
        halide_tokens = set(re.findall(r"Br|Cl|I(?![a-z])", composition))
        if len(halide_tokens) >= 2:
            risk = 0.8
            desc = (
                f"Mixed halide composition ({'/'.join(sorted(halide_tokens))}) "
                "- high risk of light-induced phase segregation"
            )
        elif len(halide_tokens) == 1:
            risk = 0.2
            desc = f"Single halide ({next(iter(halide_tokens))}) - low segregation risk"
        else:
            risk = 0.1
            desc = "No halides detected - not applicable"
        risks["halide_segregation"] = {"risk": risk, "description": desc}
        
        if "Sn" in composition:
            risk = 0.9
            desc = "Sn-based perovskite - highly susceptible to Sn²⁺ → Sn⁴⁺ oxidation"
        elif "MA" in composition:
            risk = 0.6 + (temperature - 25) / 100
            desc = f"MA-based perovskite - moderate thermal instability at {temperature}°C"
        elif "FA" in composition:
            risk = 0.4 + (temperature - 25) / 150
            desc = f"FA-based perovskite - better thermal stability at {temperature}°C"
        else:
            risk = 0.3
            desc = "Cs-based perovskite - good thermal stability"
        risks["thermal_degradation"] = {"risk": max(0.0, min(risk, 1.0)), "description": desc}
        
        if "MA" in composition:
            risk = 0.8 + (humidity - 50) / 100
            desc = f"MA⁺ is hygroscopic - high moisture sensitivity at {humidity}% RH"
        elif dimensionality == "3D":
            risk = 0.6 + (humidity - 50) / 100
            desc = f"3D perovskite - moderate moisture sensitivity at {humidity}% RH"
        elif dimensionality == "2D":
            risk = 0.3
            desc = "2D perovskite with hydrophobic spacer - improved moisture resistance"
        else:
            risk = 0.5
            desc = "Standard moisture sensitivity"
        risks["moisture_sensitivity"] = {"risk": max(0.0, min(risk, 1.0)), "description": desc}
        
        # Ion migration; same "I" pattern as halide segregation.
        if re.search(r"I(?![a-z])", composition):
            risk = 0.7
            desc = "I⁻ ions have high mobility - risk of ion migration and hysteresis"
        elif "Br" in composition:
            risk = 0.5
            desc = "Br⁻ ions have moderate mobility"
        else:
            risk = 0.3
            desc = "Cl⁻ ions have lower mobility"
        risks["ion_migration"] = {"risk": risk, "description": desc}
        
        avg_risk = sum(r["risk"] for r in risks.values()) / len(risks)
        if avg_risk < 0.4:
            overall = "Low overall degradation risk - stable material"
        elif avg_risk < 0.6:
            overall = "Moderate overall risk - requires encapsulation"
        else:
            overall = "High overall risk - advanced stability strategies needed"
        
        return {
            **risks,
            "overall_risk": overall,
            "average_risk_score": round(avg_risk, 2),
        }


class PropertyCalculator:
    """Calculate derived material properties from bandgap and structure type."""

    @classmethod
    def calculate_properties(
        cls,
        formula: str,
        bandgap: Optional[float] = None,
        structure_type: str = "Perovskite",
    ) -> Dict[str, Any]:
        """Return derived properties for display as {"properties": [...], "formula": str}."""
        properties = []

        if bandgap is not None and bandgap > 0:
            # λ_edge = hc/Eg ≈ 1240/Eg nm. Undefined for Eg ≤ 0 (metals / erroneous input).
            properties.append({
                "property": "Absorption Edge",
                "value": f"{round(1240 / bandgap, 1)} nm",
                "category": "Optical",
            })
            # Practical Voc estimate: Voc ≈ Eg/q − 0.4 V typical loss. A
            # heuristic, not the Shockley-Queisser limit. Below Eg = 0.4 eV the
            # estimate is reported as ~0 V.
            voc_max = bandgap - 0.4
            voc_display = (
                f"{round(voc_max, 2)} V" if voc_max > 0
                else "~0 V (bandgap too small for a useful photovoltage)"
            )
            properties.append({
                "property": "Practical Voc estimate (Eg − 0.4 V loss)",
                "value": voc_display,
                "category": "Device",
            })
            properties.append({
                "property": "Band Type",
                "value": "Direct" if structure_type == "Perovskite" else "Unknown",
                "category": "Electronic",
            })

        if "Perovskite" in structure_type:
            properties.append({
                "property": "Crystal Structure",
                "value": "ABX3 Perovskite",
                "category": "Structural",
            })
            properties.append({
                "property": "Coordination",
                "value": "BX6 octahedral",
                "category": "Structural",
            })

        return {"properties": properties, "formula": formula}
