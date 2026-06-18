import json


def find_json_objects(text: str) -> list[str]:
    candidates = []
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:index + 1])
    return candidates


def require_schema(model_cls):
    required_fields = [name for name, field in model_cls.model_fields.items() if field.is_required()]

    def guardrail(task_output):
        candidates = find_json_objects(task_output.raw or "")
        if not candidates:
            return False, f"No JSON object found in your answer. Respond with a single JSON object containing exactly these fields: {required_fields}."

        last_error = None
        last_data = None
        for candidate in reversed(candidates):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as error:
                last_error = error
                continue
            missing = [field for field in required_fields if field not in data]
            if not missing:
                return True, candidate
            last_data = data

        if last_data is not None:
            missing = [field for field in required_fields if field not in last_data]
            return False, (
                f"Your output is missing required fields {missing}. You produced an object "
                f"with fields {list(last_data.keys())} instead — that looks like a different task's "
                f"output. This task must return a NEW object with exactly these fields: {required_fields}."
            )
        return False, f"Your JSON is malformed ({last_error}). Respond with a single valid JSON object containing exactly these fields: {required_fields}."

    return guardrail
