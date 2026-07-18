from __future__ import annotations

import re
from typing import Dict, List, Optional

HAND_ONLY_BANK: Dict[str, str] = {
    "Ability": "A white robotic hand with a rounded shell, dark finger coverings, and a thick side thumb close to the palm plane.",
    "Allegro": "A dark modular robotic hand with a blocky palm and slim segmented fingers ending in bright fingertip caps.",
    "DexHand021": "A silver robotic hand with exposed mechanisms, a ribbed palm, and slender articulated fingers.",
    "Leap": "A compact low-profile hand with a blocky palm and short rectangular finger segments.",
    "Wuji": "A slim dexterous robotic hand with long narrow fingers, light metallic finger links, and a compact dark palm base.",
    "OrcaHand": "A dark robotic hand with a narrow palm and five slender fingers spread in a fan-like configuration.",
    "Revo2": "A smooth anthropomorphic robotic hand with a rounded palm and slim fingers with softly curved matte-gray surfaces.",
    "RH5DG2": "A white-and-gray robotic hand with a rounded palm shell, dark cylindrical finger coverings, and a thick side thumb.",
    "RH56DFX": "A white-and-silver robotic hand with a sculpted palm shell, black finger pads, and a diagonal striped panel on the back.",
    "RoHand": "A white robotic hand with an enclosed palm shell, dark metallic finger segments, and a side-mounted thumb.",
    "Schunk SVH": "A white dexterous hand with gray fingertip pads, robust finger links, and a dark central palm pad.",
    "Shadow Hand": "A black anthropomorphic hand with slim multi-joint fingers, white joint markings, and long human-like proportions.",
    "Sharpa": "A matte gray robotic hand with a clean anthropomorphic silhouette, smooth palm surfaces, and simple segmented fingers.",
}

HAND_ARM_BANK: Dict[str, str] = {
    "Jaka Zu7 + DexHand021": "A metallic-gray collaborative arm with blue joint caps, paired with a silver dexterous hand with dense exposed mechanisms.",
    "KUKA + Sharpa": "A white KUKA-style arm with orange accent panels, paired with a smooth anthropomorphic hand.",
    "Panda + Allegro": "A light Panda-style arm paired with a compact dark modular hand with rectangular segmented fingers.",
    "Panda + Orca": "A light Panda-style arm attached to a dark hand with a narrow palm and strongly spread fingers.",
    "RM65 + BrainCo": "A long white RM_65 robotic arm with smooth enclosed links and rounded elbow sections, ending in a compact BrainCo anthropomorphic hand with four close fingers and an opposable thumb.",
    "UR5 + RH56DFX": "A metallic-gray industrial arm with blue joint caps, ending in a white-and-silver dexterous hand with a sculpted palm shell.",
    "UR5 + RH5DG2": "A metallic-gray UR5-style arm attached to a white-and-gray hand with a rounded palm shell and dark cylindrical finger coverings.",
    "UR5 + Schunk Hand": "A metallic-gray arm with blue round joint covers, paired with a robust white industrial hand with thick segmented fingers.",
    "UR5 + Shadow Hand": "A metallic-gray arm with blue circular joints, attached to a dark anthropomorphic hand with slim multi-joint fingers.",
    "UR5 + Wuji": "A metallic-gray arm with blue circular joint caps, ending in a slim dexterous hand with long narrow fingers.",
    "xArm + Ability": "A white arm with smooth enclosed links, paired with a compact white hand with a rounded palm shell and dark finger coverings.",
    "xArm + Leap": "A white arm with smooth rounded links, attached to a compact low-profile hand with a blocky palm and modular fingers.",
    "RM75 + RoHand": "A white collaborative arm with dark joints, paired with a smooth white robotic hand with an enclosed palm shell, dark metallic finger segments, and a side-mounted thumb.",
}

_ALIAS_TO_CANONICAL = {
    "ability": "Ability",
    "allegro": "Allegro",
    "dexhand021": "DexHand021",
    "inspire": "RH56DFX",  # Hand-only Inspire variants are evaluated under RH56DFX.
    "leap": "Leap",
    "orca": "OrcaHand",
    "orcahand": "OrcaHand",
    "revo2": "Revo2",
    "rh5dg2": "RH5DG2",
    "rh56dfx": "RH56DFX",
    "rohand": "RoHand",
    "schunksvh": "Schunk SVH",
    "shadow": "Shadow Hand",
    "shadowhand": "Shadow Hand",
    "sharpa": "Sharpa",
    "jakazu7dexhand021": "Jaka Zu7 + DexHand021",
    "kukasharpa": "KUKA + Sharpa",
    "pandaallegro": "Panda + Allegro",
    "pandaorca": "Panda + Orca",
    "rm65brainco": "RM65 + BrainCo",
    "ur5rh56dfx": "UR5 + RH56DFX",
    "ur5rh5dg2": "UR5 + RH5DG2",
    "ur5schunkhand": "UR5 + Schunk Hand",
    "ur5shadowhand": "UR5 + Shadow Hand",
    "ur5wuji": "UR5 + Wuji",
    "xarmability": "xArm + Ability",
    "xarmleap": "xArm + Leap",
    "rm75rohand": "RM75 + RoHand",
    "brainco": "RM65 + BrainCo",
    "wuji": "Wuji",
}

DEFAULT_TRACK = "urdf_reference_conditioned"


def normalize_scope(scope: str) -> str:
    text = (scope or "").lower().replace("_", "-").strip()
    if text in {"hand", "hand-only", "handonly"}:
        return "hand-only"
    if text in {"hand-arm", "handarm", "arm-hand"}:
        return "hand-arm"
    return text

def _normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def infer_group(target_name: str = "", replacement_scope: str = "") -> Optional[str]:
    scope = normalize_scope(replacement_scope)
    if scope == "hand-only":
        return "hand_only"
    if scope == "hand-arm":
        return "hand_arm"
    canonical = canonicalize_target_name(target_name)
    if canonical in HAND_ONLY_BANK:
        return "hand_only"
    if canonical in HAND_ARM_BANK:
        return "hand_arm"
    return None


def get_bank(group: Optional[str] = None) -> Dict[str, str]:
    if group == "hand_only":
        return dict(HAND_ONLY_BANK)
    if group == "hand_arm":
        return dict(HAND_ARM_BANK)
    both = dict(HAND_ONLY_BANK)
    both.update(HAND_ARM_BANK)
    return both


def canonicalize_target_name(name: str) -> str:
    if not name:
        return ""
    if name in HAND_ONLY_BANK or name in HAND_ARM_BANK:
        return name
    normalized = _normalize_name(name)
    if normalized in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[normalized]
    for candidate in list(HAND_ONLY_BANK.keys()) + list(HAND_ARM_BANK.keys()):
        if _normalize_name(candidate) == normalized:
            return candidate
    return name.strip()


def description_for_target(target_name: str, replacement_scope: str = "") -> str:
    canonical = canonicalize_target_name(target_name)
    group = infer_group(canonical, replacement_scope)
    return get_bank(group).get(canonical, "")


def candidate_descriptions_for_group(replacement_scope: str = "", target_name: str = "") -> List[str]:
    group = infer_group(target_name, replacement_scope)
    return list(get_bank(group).values())


def compact_descriptor(full_description: str, max_words: int = 18) -> str:
    if not full_description:
        return ""
    first_sentence = full_description.split(".", 1)[0].strip()
    words = first_sentence.split()
    if len(words) <= max_words:
        return first_sentence
    return " ".join(words[:max_words]).rstrip(",;:")


def build_instruction(replacement_scope: str, target_name: str, descriptor: str) -> str:
    normalized_scope = normalize_scope(replacement_scope)
    region = "hand-arm" if normalized_scope == "hand-arm" else "hand"
    return (
        f"Replace the human {region} region in this egocentric interaction image with the dexterous robot shown in the reference image. "
        f"The target robot is {target_name}, described as: {descriptor}. "
        "Preserve the original hand-object interaction, object state, and surrounding scene structure as much as possible."
    )


def enrich_record(record: dict) -> dict:
    enriched = dict(record)
    target_name = enriched.get("target_name", "") or enriched.get("robot_name", "")
    scope = normalize_scope(enriched.get("replacement_scope", "") or enriched.get("scope", ""))
    if scope:
        enriched["replacement_scope"] = scope

    canonical = canonicalize_target_name(str(target_name))
    if canonical:
        enriched["target_name"] = canonical

    full_description = (
        enriched.get("target_description", "")
        or enriched.get("embodiment_description", "")
        or description_for_target(canonical, scope)
    )
    if full_description:
        enriched["target_description"] = full_description

    if not enriched.get("candidate_descriptions"):
        candidates = candidate_descriptions_for_group(scope, canonical)
        if candidates:
            enriched["candidate_descriptions"] = candidates

    if canonical and not enriched.get("compact_descriptor"):
        enriched["compact_descriptor"] = compact_descriptor(full_description)

    if canonical and not enriched.get("instruction"):
        enriched["instruction"] = build_instruction(scope or "hand-only", canonical, enriched.get("compact_descriptor", ""))

    return enriched
