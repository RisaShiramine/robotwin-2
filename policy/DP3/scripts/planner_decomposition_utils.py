from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

TABLE_SUPPORTS = {None, "table", "center", "table_center"}


def dedupe_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    result = []
    for item in items:
        if item in (None, "", []):
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def is_table_support(value: Any) -> bool:
    return value in TABLE_SUPPORTS


def infer_stage_type(stage: Dict[str, Any]) -> str:
    if stage.get("stage_type"):
        return stage["stage_type"]
    target_support = stage.get("target_support")
    target_location = stage.get("target_location")
    action_type = stage.get("action_type")
    if action_type == "stack" or (target_support and not is_table_support(target_support)):
        return "stack_on_support"
    if target_location:
        return "move_to_region"
    return action_type or "other"


def infer_source_object(stage: Dict[str, Any]) -> Optional[str]:
    return stage.get("source_object") or stage.get("target_object")


def infer_support_object(stage: Dict[str, Any]) -> Optional[str]:
    support = stage.get("support_object") or stage.get("target_support")
    return None if is_table_support(support) else support


def infer_target_region(stage: Dict[str, Any]) -> Optional[str]:
    return stage.get("target_region") or stage.get("target_location")


def build_stage_label(stage: Dict[str, Any]) -> str:
    source_object = infer_source_object(stage)
    support_object = infer_support_object(stage)
    target_region = infer_target_region(stage)
    if support_object:
        return f"{source_object} on {support_object}"
    if target_region:
        return f"{source_object} at {target_region}"
    action_type = stage.get("action_type") or stage.get("stage_type") or "stage"
    return f"{action_type} {source_object}".strip()


def topologically_order_stack_stages(stack_stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = []
    placed = set()
    remaining = [dict(stage) for stage in stack_stages]
    while remaining:
        progress = False
        for stage in list(remaining):
            support = infer_support_object(stage)
            if support is None or support in placed:
                ordered.append(stage)
                placed.add(infer_source_object(stage))
                remaining.remove(stage)
                progress = True
        if not progress:
            ordered.extend(remaining)
            break
    return ordered


def _sanitize_name(value: Any) -> str:
    return str(value).replace(" ", "_")


def _build_execution_stage(
    stage_name: str,
    stage_type: str,
    action_type: str,
    source_object: Any,
    target_object: Any,
    support_object: Any,
    target_support: Any,
    target_region: Any,
    preferred_arm: Any,
    required_objects: List[Any],
    success_criteria: str,
    target_location: Any = None,
    spatial_relation: Any = None,
) -> Dict[str, Any]:
    return {
        "stage": stage_name,
        "stage_type": stage_type,
        "action_type": action_type,
        "source_object": source_object,
        "target_object": target_object,
        "support_object": support_object,
        "target_support": target_support,
        "target_region": target_region,
        "target_location": target_location,
        "spatial_relation": spatial_relation,
        "preferred_arm": preferred_arm,
        "required_objects": dedupe_keep_order(required_objects),
        "completed_subgoals_before_stage": [],
        "success_criteria": success_criteria,
    }


def _find_source_stage(stage_lookup: Dict[Any, Dict[str, Any]], source_object: Any) -> Dict[str, Any]:
    stage = stage_lookup.get(source_object)
    if stage is not None:
        return stage
    return {
        "preferred_arm": None,
        "target_region": "center",
        "target_location": "center",
    }


def _build_stack_execution_stages(stages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    stack_stages = [dict(stage) for stage in stages if infer_support_object(stage) is not None]
    if not stack_stages:
        return None

    ordered_stack_stages = topologically_order_stack_stages(stack_stages)
    base_object = infer_support_object(ordered_stack_stages[0])
    move_like_stage_by_object = {
        infer_source_object(stage): dict(stage)
        for stage in stages
        if infer_source_object(stage) is not None and infer_support_object(stage) is None
    }

    placement_plan: List[Tuple[Any, Any, Dict[str, Any]]] = []
    base_source_stage = _find_source_stage(move_like_stage_by_object, base_object)
    placement_plan.append((base_object, None, base_source_stage))
    for stack_stage in ordered_stack_stages:
        placement_plan.append((infer_source_object(stack_stage), infer_support_object(stack_stage), dict(stack_stage)))

    execution_stages: List[Dict[str, Any]] = []
    for order_idx, (source_object, support_object, raw_stage) in enumerate(placement_plan, start=1):
        region = infer_target_region(raw_stage) or "center"
        preferred_arm = raw_stage.get("preferred_arm")
        if preferred_arm is None:
            preferred_arm = move_like_stage_by_object.get(source_object, {}).get("preferred_arm")
        support_label = support_object or "table"
        target_object = support_object or source_object
        target_support = support_object or "table"
        required_objects = [source_object, support_object] if support_object else [source_object]
        source_key = _sanitize_name(source_object)
        support_key = _sanitize_name(support_label)

        execution_stages.extend(
            [
                _build_execution_stage(
                    stage_name=f"align_{source_key}_step_{order_idx}",
                    stage_type="align_for_pick",
                    action_type="align",
                    source_object=source_object,
                    target_object=target_object,
                    support_object=support_object,
                    target_support=target_support,
                    target_region=region,
                    target_location=region,
                    preferred_arm=preferred_arm,
                    required_objects=required_objects,
                    success_criteria=f"Arm is initialized in free space and aligned above {source_object}.",
                ),
                _build_execution_stage(
                    stage_name=f"grasp_{source_key}_step_{order_idx}",
                    stage_type="grasp_object",
                    action_type="pick",
                    source_object=source_object,
                    target_object=source_object,
                    support_object=support_object,
                    target_support=target_support,
                    target_region=region,
                    target_location=region,
                    preferred_arm=preferred_arm,
                    required_objects=required_objects,
                    success_criteria=f"Arm grasps {source_object} securely.",
                ),
                _build_execution_stage(
                    stage_name=f"transport_{source_key}_to_{support_key}_step_{order_idx}",
                    stage_type="transport_in_air",
                    action_type="move",
                    source_object=source_object,
                    target_object=target_object,
                    support_object=support_object,
                    target_support=target_support,
                    target_region=region,
                    target_location=region,
                    preferred_arm=preferred_arm,
                    required_objects=required_objects,
                    success_criteria=f"{source_object} is transported in the air toward {region}.",
                    spatial_relation="over",
                ),
                _build_execution_stage(
                    stage_name=f"place_release_{source_key}_on_{support_key}_step_{order_idx}",
                    stage_type="place_release",
                    action_type="place",
                    source_object=source_object,
                    target_object=target_object,
                    support_object=support_object,
                    target_support=target_support,
                    target_region=region,
                    target_location=region,
                    preferred_arm=preferred_arm,
                    required_objects=required_objects,
                    success_criteria=(
                        f"Arm descends, releases {source_object}, and the block is stably placed "
                        f"on {support_label} in {region}."
                    ),
                    spatial_relation="over" if support_object else "on",
                ),
                _build_execution_stage(
                    stage_name=f"retreat_reset_{source_key}_step_{order_idx}",
                    stage_type="retreat_and_reset",
                    action_type="reset",
                    source_object=source_object,
                    target_object=target_object,
                    support_object=support_object,
                    target_support=target_support,
                    target_region=region,
                    target_location=region,
                    preferred_arm=preferred_arm,
                    required_objects=required_objects,
                    success_criteria=f"Arm lifts away from the tower and resets after placing {source_object}.",
                ),
            ]
        )

    return execution_stages


def normalize_decomposition_row(row: Dict[str, Any], expand_stack_execution: bool = True) -> Dict[str, Any]:
    decomposition = row["decomposition"]
    stages = decomposition.get("stages", [])
    if len(stages) <= 1:
        return row

    if expand_stack_execution:
        expanded_stages = _build_stack_execution_stages(stages)
        if expanded_stages:
            normalized_stages = expanded_stages
        else:
            normalized_stages = [dict(stage) for stage in stages]
    else:
        normalized_stages = [dict(stage) for stage in stages]

    completed = []
    for stage in normalized_stages:
        stage.setdefault("stage_type", infer_stage_type(stage))
        stage.setdefault("source_object", infer_source_object(stage))
        stage.setdefault("support_object", infer_support_object(stage))
        stage.setdefault("target_region", infer_target_region(stage))
        stage["completed_subgoals_before_stage"] = completed[:]
        completed.append(build_stage_label(stage))

    normalized = dict(row)
    normalized["decomposition"] = dict(decomposition)
    normalized["decomposition"]["stages"] = normalized_stages
    return normalized
