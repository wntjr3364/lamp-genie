"""User-facing labels for structured evidence; never changes a scientific filter."""
import re


LABELS = {
    "data_source": ("출처", "Source"), "chemical_class": ("화학 분류", "Chemical class"),
    "dimensionality": ("차원 분류", "Dimensionality"), "structure_type": ("구조 분류", "Structure type"),
    "device": ("대상 장치", "Target device"), "carrying_property": ("기록된 물성", "Recorded property"),
    "bandgap": ("밴드갭", "band gap"), "energy_above_hull": ("hull 기준 초과 에너지", "energy above hull"),
    "lead_free": ("납 미포함", "Lead-free"), "functional_role": ("기능 분류", "Functional role"),
    "crystal_system": ("결정계", "Crystal system"), "morphology": ("형태", "Morphology"),
    "pce": ("PCE", "PCE"), "voc": ("Voc", "Voc"), "jsc": ("Jsc", "Jsc"), "ff": ("FF", "FF"),
    "Seebeck_coefficient": ("제벡 계수", "Seebeck coefficient"),
    "electrical_conductivity": ("전기 전도도", "electrical conductivity"),
    "power_factor": ("열전 파워팩터", "power factor"),
}


def label(value, ko):
    text = str(value)
    if ko:
        text = re.sub(r"\bexactly (\d+) elements?\b", r"원소 \1종", text)
        text = re.sub(r"(\d)\s+and\s+(?=[<>≤≥])", r"\1, ", text)   # ">= 3 and <= 4" -> ">= 3, <= 4"
    for key in sorted(LABELS, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(key) + r"\b", LABELS[key][0 if ko else 1], text)
    return text.replace("_", " ")


def scope_lines(filters, ko):
    out = []
    for key, value in filters.items():
        if value in (None, "", [], {}):
            continue
        name = label(key, ko)
        if isinstance(value, bool):
            out.append(name if value else (("제외: " if ko else "Excluded: ") + name))
        else:
            values = value if isinstance(value, (list, tuple)) else [value]
            out.append(f"{name}: " + ", ".join(str(v) for v in values))
    return out


def note(text, ko):
    """Keep caveats visible without console-style capitalization."""
    text = str(text)
    if text.startswith("passing the platform's screen"):
        return ("100 mW/cm² 조명, 네 지표 양수, PCE 상한과 J–V 지표 일치 범위(2배 이내)를 통과한 기록"
                if ko else "one-sun records with all four positive metrics, within the PCE ceiling and a factor of two of their J–V product")
    if "NOT MEASUREMENTS:" in text:
        return ("조성으로부터 할당된 밴드갭은 측정값 집계에서 제외했습니다. 상세 제외 근거는 도구 결과에 보존되어 있습니다."
                if ko else "Composition-assigned band gaps are excluded from the measurement counts. Detailed exclusion evidence is retained with the tool result.")
    text = re.sub(r"\bNOT\b", "not", text)
    text = re.sub(r"(\d)\s+fraction\b", r"\1", text)  # a fill factor is a fraction without a unit
    # The plan merges earlier turns' conditions, so a condition may come from an earlier question.
    text = text.replace("(from the question)", "(대화에서 지정)" if ko else "(from the conversation)")
    return text


GROUP_REQUEST = re.compile(
    r"\b(?:by|per|each|every)\s+(?:data\s+)?(?:source|class|group|dimension|property|crystal)"
    r"|\b(?:which|compare|comparison|breakdown|distribution)\b|별로|별\s*(?:개수|분포|집계)|출처별|소스별|분류별", re.I)
COUNT_DEFINITION = re.compile(
    r"\b(?:previous|previously|earlier|before|definition|definitions|counted|recount|difference|different)\b"
    r"|이전|전에|차이|기준|다르|다른\s*개수", re.I)


def is_nitride_count_comparison(question):
    """A count-definition comparison, not a request to recommend materials."""
    return bool(COUNT_DEFINITION.search(question)
                and re.search(r"\bcounts?\b|\bnumber\b|집계|개수", question, re.I)
                and re.search(r"\bnitrides?\b|질화물", question, re.I)
                and re.search(r"\bclass\b|classification|분류", question, re.I)
                and re.search(r"nitrogen[-\s]+containing|\bN[-\s]+containing|질소.*포함|N.*포함", question, re.I)
                and not re.search(r"\band\s+(?:explain|why|recommend|predict|calculate)\b|그리고\s*(?:설명|추천|예측|계산)", question, re.I))
