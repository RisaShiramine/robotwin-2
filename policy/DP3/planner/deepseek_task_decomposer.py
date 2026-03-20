import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

SYSTEM_PROMPT = """
You are a robot task decomposition engine for tabletop manipulation.
Convert each natural-language task instruction into a normalized JSON object for a planner.

Requirements:
1. Output valid JSON only.
2. Preserve the user instruction exactly in `instruction`.
3. Infer a canonical task description shared across paraphrases.
4. Break the task into ordered planner stages.
5. Use concise stage names such as `move_red_to_center`, `place_green_on_red`.
6. For each stage, identify:
   - stage
   - stage_type
   - action_type
   - source_object
   - target_object
   - support_object
   - target_support
   - target_region
   - target_location
   - spatial_relation
   - required_objects
   - completed_subgoals_before_stage
   - success_criteria
7. If an arm is mentioned, store it in `preferred_arm`; otherwise use null.
8. Include `scene_objects` as a deduplicated list.
9. Include `final_goal` as a short sentence.
10. Include `task_id` if a stable task identifier is available; otherwise use null.
11. If something is absent, use null or an empty list instead of inventing details.

Return exactly this schema:
{
  "instruction": "original instruction",
  "canonical_task": "normalized task description",
  "task_id": "stable task id or null",
  "task_category": "stack|sort|place|pick_place|other",
  "scene_objects": ["..."],
  "final_goal": "...",
  "stages": [
    {
      "stage": "...",
      "stage_type": "establish_base|move_to_region|stack_on_support|pick_place|other",
      "action_type": "move|pick|place|stack|other",
      "source_object": "...",
      "target_object": "...",
      "support_object": "...",
      "target_support": "...",
      "target_region": "...",
      "target_location": "...",
      "spatial_relation": "...",
      "preferred_arm": "left|right|both|null",
      "required_objects": ["..."],
      "completed_subgoals_before_stage": ["..."],
      "success_criteria": "..."
    }
  ]
}
""".strip()


@dataclass
class DeepSeekTaskDecomposer:
    api_key: str
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 60
    max_completion_tokens: int = 1200
    retry_backoff_max: int = 16

    def __post_init__(self) -> None:
        self._build_client()

    def _build_client(self) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "openai is required to use the DeepSeek task decomposition workflow. "
                "Install it with `python -m pip install openai`."
            ) from exc
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def decompose_instruction(self, instruction: str) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    max_tokens=self.max_completion_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Decompose this instruction into the required JSON schema.\n"
                                f"Instruction: {instruction}"
                            ),
                        },
                    ],
                )
                content = completion.choices[0].message.content or ""
                parsed = _extract_json(content)
                _validate_payload(parsed, instruction)
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_retries:
                    break
                self._build_client()
                time.sleep(min(2**attempt, self.retry_backoff_max))
        raise RuntimeError(f"Failed to decompose instruction after {self.max_retries} attempts: {instruction}") from last_error


def decompose_instruction_batch(
    instructions: Iterable[str],
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    temperature: float = 0.0,
    max_retries: int = 3,
    timeout: int = 60,
    max_completion_tokens: int = 1200,
    retry_backoff_max: int = 16,
) -> List[Dict[str, Any]]:
    decomposer = DeepSeekTaskDecomposer(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        max_completion_tokens=max_completion_tokens,
        retry_backoff_max=retry_backoff_max,
    )
    return [decomposer.decompose_instruction(instruction) for instruction in instructions]


def load_api_key(env_name: str = "DEEPSEEK_API_KEY") -> str:
    api_key = os.environ.get(env_name)
    if not api_key:
        raise EnvironmentError(f"Environment variable {env_name} is not set.")
    return api_key


def _extract_json(content: str) -> Dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _validate_payload(payload: Dict[str, Any], instruction: str) -> None:
    required_keys = {
        "instruction",
        "canonical_task",
        "task_id",
        "task_category",
        "scene_objects",
        "final_goal",
        "stages",
    }
    missing = required_keys - payload.keys()
    if missing:
        raise ValueError(f"Missing keys in response: {sorted(missing)}")
    if payload["instruction"] != instruction:
        raise ValueError("Returned instruction does not match input instruction.")
    if not isinstance(payload["stages"], list) or not payload["stages"]:
        raise ValueError("Response must contain at least one stage.")
