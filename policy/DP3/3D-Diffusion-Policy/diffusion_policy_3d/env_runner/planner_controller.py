import json
from pathlib import Path


class PlannerTokenStateMachine:
    """
    Lightweight rule-based wrapper that converts decomposed long-horizon stages
    into discrete planner token IDs for DP3 inference.
    """

    def __init__(self, vocab_path, stages, completion_rule=None, debug=False, debug_prefix="[PlannerTokenStateMachine]"):
        self.vocab_path = Path(vocab_path)
        with self.vocab_path.open("r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.stages = list(stages)
        self.completion_rule = completion_rule or (lambda obs, stage: False)
        self.current_stage_idx = 0
        self.debug = debug
        self.debug_prefix = debug_prefix
        self._debug(
            f"initialized with {len(self.stages)} stages; current={self.describe_stage(self.current_stage_idx)}"
        )

    def _debug(self, message):
        if self.debug:
            print(f"{self.debug_prefix} {message}")

    def _lookup_stage_id(self, stage_value):
        normalized = str(stage_value).lower().strip()
        stage_id = self.vocab["stage"].get(normalized, 0)
        if stage_id == 0 and normalized != "unknown":
            raise ValueError(f"Planner stage token '{stage_value}' is missing from planner_vocab.json")
        return stage_id

    def _lookup_object_id(self, object_value):
        if object_value is None:
            return self.vocab["object"].get("null", 1)
        normalized = str(object_value).lower().strip()
        if not normalized:
            return self.vocab["object"].get("null", 1)
        object_id = self.vocab["object"].get(normalized, 0)
        if object_id == 0 and normalized not in {"unknown", "null"}:
            raise ValueError(f"Planner object token '{object_value}' is missing from planner_vocab.json")
        return object_id

    def reset(self):
        self.current_stage_idx = 0
        self._debug(f"reset -> {self.describe_stage(self.current_stage_idx)}")

    @property
    def current_stage(self):
        return self.stages[self.current_stage_idx]

    def describe_stage(self, stage_idx=None):
        if not self.stages:
            return "<empty>"
        if stage_idx is None:
            stage_idx = self.current_stage_idx
        stage = self.stages[stage_idx]
        stage_type = stage.get("runtime_stage_type") or stage.get("stage_type", "unknown")
        source = stage.get("source_object", "unknown")
        target = (
            stage.get("runtime_target_object")
            or stage.get("runtime_target_region")
            or stage.get("runtime_target_support")
            or stage.get("target_object")
            or stage.get("target_region")
            or stage.get("target_support")
            or "null"
        )
        return f"stage[{stage_idx}] type={stage_type}, source={source}, target={target}"

    def maybe_advance(self, obs):
        if self.current_stage_idx >= len(self.stages) - 1:
            return False
        if self.completion_rule(obs, self.current_stage):
            prev_stage_idx = self.current_stage_idx
            self.current_stage_idx += 1
            self._debug(
                f"advance: {self.describe_stage(prev_stage_idx)} -> {self.describe_stage(self.current_stage_idx)}"
            )
            return True
        return False

    def current_tokens(self):
        stage = self.current_stage
        token_stage_type = stage.get("token_stage_type") or stage.get("stage_type", "unknown")
        token_source_object = stage.get("token_source_object") or stage.get("source_object", "unknown")
        target_value = (
            stage.get("token_target_object")
            or stage.get("token_target_region")
            or stage.get("token_target_support")
            or stage.get("target_object")
            or stage.get("target_region")
            or stage.get("target_support")
        )
        stage_id = self._lookup_stage_id(token_stage_type)
        source_id = self._lookup_object_id(token_source_object)
        target_id = self._lookup_object_id(target_value)
        return {
            "stage_id": stage_id,
            "source_id": source_id,
            "target_id": target_id,
        }
