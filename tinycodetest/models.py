from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tinycodetest.schema import Task


class ModelAdapter(Protocol):
    name: str

    def generate(self, *, prompt: str, task: Task, k: int, seed: int) -> list[str]:
        ...


def _post_json_request(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    model_label: str,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{model_label} HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{model_label} request failed: {exc}") from exc

    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{model_label} returned a non-JSON response.") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"{model_label} returned a non-object JSON response.")
    return raw


def _build_code(signature: str, body: str) -> str:
    lines = [signature]
    for line in body.strip("\n").splitlines():
        lines.append(f"    {line}" if line else "")
    return "\n".join(lines).rstrip() + "\n"


def _extract_int(prompt: str, pattern: str, default: int) -> int:
    match = re.search(pattern, prompt)
    if not match:
        return default
    try:
        return int(match.group(1))
    except Exception:
        return default


def _extract_char(prompt: str) -> str:
    match = re.search(r"character\s+(.+?)\s+in text", prompt)
    if not match:
        return "a"
    raw = match.group(1).strip()
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return "a"
    if isinstance(parsed, str) and len(parsed) == 1:
        return parsed
    return "a"


def _template(task: Task) -> str:
    return str((task.metadata or {}).get("template", ""))


def _fallback(signature: str) -> str:
    return _build_code(signature, "raise NotImplementedError")


def _small_solution(task: Task, prompt: str) -> str:
    template = _template(task)
    if template == "add_constant":
        delta = _extract_int(prompt, r"constant of\s+(-?\d+)", 0)
        return _build_code(task.signature, f"return x + ({delta})")

    if template == "count_char":
        target = _extract_char(prompt)
        return _build_code(task.signature, f"return text.count({target!r})")

    if template == "clamp":
        lo = _extract_int(prompt, r"\[(-?\d+),\s*-?\d+\]", -1)
        hi = _extract_int(prompt, r"\[-?\d+,\s*(-?\d+)\]", 1)
        return _build_code(
            task.signature,
            "\n".join(
                [
                    f"if x < {lo}:",
                    f"    return {lo}",
                    f"if x > {hi}:",
                    f"    return {hi}",
                    "return x",
                ]
            ),
        )

    if template == "sum_even":
        return _build_code(task.signature, "return sum(v for v in nums if v % 2 == 0)")

    if template == "reverse_words":
        return _build_code(task.signature, 'return " ".join(text.split(" ")[::-1])')

    return _fallback(task.signature)


def _medium_solution(task: Task, prompt: str) -> str:
    template = _template(task)
    if template in {"add_constant", "count_char", "clamp", "sum_even"}:
        return _small_solution(task, prompt)

    if template == "reverse_words":
        return _build_code(task.signature, 'return " ".join(reversed(text.split()))')

    if template == "first_unique":
        return _build_code(
            task.signature,
            "\n".join(
                [
                    "counts = {}",
                    "for ch in text:",
                    "    counts[ch] = counts.get(ch, 0) + 1",
                    "for idx, ch in enumerate(text):",
                    "    if counts[ch] == 1:",
                    "        return idx",
                    "return -1",
                ]
            ),
        )

    if template == "rotate_list":
        k = _extract_int(prompt, r"right by\s+(\d+)\s+positions", 1)
        return _build_code(
            task.signature,
            "\n".join(
                [
                    "if not nums:",
                    "    return []",
                    f"shift = {k} % len(nums)",
                    "return nums[-shift:] + nums[:-shift]",
                ]
            ),
        )

    if template == "valid_brackets":
        return _build_code(
            task.signature,
            "\n".join(
                [
                    "stack = []",
                    'pairs = {")": "(", "]": "[", "}": "{"}',
                    "for ch in text:",
                    "    if ch in '([{':",
                    "        stack.append(ch)",
                    "    elif ch in ')]}':",
                    "        if not stack or stack[-1] != pairs[ch]:",
                    "            return False",
                    "        stack.pop()",
                    "    else:",
                    "        return False",
                    "return not stack",
                ]
            ),
        )

    if template == "longest_common_prefix":
        return _build_code(
            task.signature,
            "\n".join(
                [
                    "if not words:",
                    "    return ''",
                    "prefix = words[0]",
                    "for word in words[1:]:",
                    "    while not word.startswith(prefix):",
                    "        prefix = prefix[:-1]",
                    "        if not prefix:",
                    "            return ''",
                    "return prefix",
                ]
            ),
        )

    if template == "group_anagrams":
        return _build_code(
            task.signature,
            "\n".join(
                [
                    "groups = {}",
                    "for word in words:",
                    "    key = ''.join(sorted(word))",
                    "    groups.setdefault(key, []).append(word)",
                    "out = [sorted(items) for items in groups.values()]",
                    "out.sort(key=lambda grp: grp[0] if grp else '')",
                    "return out",
                ]
            ),
        )

    if template == "coin_change":
        return _build_code(
            task.signature,
            "\n".join(
                [
                    "limit = amount + 1",
                    "dp = [limit] * (amount + 1)",
                    "dp[0] = 0",
                    "for value in range(1, amount + 1):",
                    "    for coin in coins:",
                    "        if coin <= value:",
                    "            dp[value] = min(dp[value], dp[value - coin] + 1)",
                    "return dp[amount] if dp[amount] != limit else -1",
                ]
            ),
        )

    if template == "lis_length":
        return _build_code(
            task.signature,
            "\n".join(
                [
                    "if not nums:",
                    "    return 0",
                    "tails = []",
                    "from bisect import bisect_left",
                    "for value in nums:",
                    "    idx = bisect_left(tails, value)",
                    "    if idx == len(tails):",
                    "        tails.append(value)",
                    "    else:",
                    "        tails[idx] = value",
                    "return len(tails)",
                ]
            ),
        )

    return _fallback(task.signature)


def _degrade(code: str) -> str:
    if "return" not in code:
        return code
    lines = code.splitlines()
    out: list[str] = []
    patched = False
    for line in lines:
        if not patched and line.strip().startswith("return"):
            out.append("    return None")
            patched = True
        else:
            out.append(line)
    return "\n".join(out) + "\n"


@dataclass
class HeuristicSmallModel:
    name: str = "heuristic-small"
    degrade_probability: float = 0.35

    def generate(self, *, prompt: str, task: Task, k: int, seed: int) -> list[str]:
        rng = random.Random(seed)
        base = _small_solution(task, prompt)
        outputs: list[str] = []
        for idx in range(k):
            if idx > 0 and rng.random() < self.degrade_probability:
                outputs.append(_degrade(base))
            else:
                outputs.append(base)
        return outputs


@dataclass
class HeuristicMediumModel:
    name: str = "heuristic-medium"
    degrade_probability: float = 0.2

    def generate(self, *, prompt: str, task: Task, k: int, seed: int) -> list[str]:
        rng = random.Random(seed)
        base = _medium_solution(task, prompt)
        outputs: list[str] = []
        for idx in range(k):
            if idx > 0 and rng.random() < self.degrade_probability:
                outputs.append(_degrade(base))
            else:
                outputs.append(base)
        return outputs


@dataclass
class ReferenceOracleModel:
    name: str = "reference-oracle"

    def generate(self, *, prompt: str, task: Task, k: int, seed: int) -> list[str]:
        _ = (prompt, seed)
        return [task.canonical_solution for _ in range(k)]


@dataclass
class OpenAICompatibleModel:
    name: str = "openai-compatible"
    model_name: str = ""
    api_base: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 512
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name:
            self.model_name = os.getenv("TCT_OPENAI_MODEL", "")
        if not self.api_base:
            self.api_base = os.getenv("TCT_OPENAI_BASE_URL", "https://api.openai.com/v1")

    def _request_completion(self, prompt: str, seed: int) -> str:
        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for model `{self.name}`.")
        if not self.model_name:
            raise RuntimeError(
                f"No model_name configured for `{self.name}`. Set TCT_OPENAI_MODEL or model config."
            )

        url = f"{self.api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": seed,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only Python code implementing the requested function.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        raw = _post_json_request(
            url=url,
            payload=payload,
            headers=headers,
            model_label=self.name,
            timeout_seconds=60,
        )
        choices = raw.get("choices", [])
        if not choices:
            raise RuntimeError(f"{self.name} response had no choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"{self.name} response content was empty.")
        return content

    def generate(self, *, prompt: str, task: Task, k: int, seed: int) -> list[str]:
        _ = task
        outputs: list[str] = []
        for idx in range(k):
            outputs.append(self._request_completion(prompt, seed + idx))
        return outputs


@dataclass
class AnthropicModel:
    name: str = "anthropic"
    model_name: str = ""
    api_base: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 1024
    anthropic_version: str = "2023-06-01"
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name:
            self.model_name = os.getenv("TCT_ANTHROPIC_MODEL", "")

    def _request_completion(self, prompt: str) -> str:
        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for model `{self.name}`.")
        if not self.model_name:
            raise RuntimeError(
                f"No model_name configured for `{self.name}`. Set TCT_ANTHROPIC_MODEL or model config."
            )

        url = f"{self.api_base.rstrip('/')}/messages"
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": "Return only Python code implementing the requested function.",
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        headers.update(self.extra_headers)

        raw = _post_json_request(
            url=url,
            payload=payload,
            headers=headers,
            model_label=self.name,
            timeout_seconds=60,
        )
        content = raw.get("content", [])
        if not isinstance(content, list) or not content:
            raise RuntimeError(f"{self.name} response had no content blocks.")

        chunks: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
        combined = "\n".join(chunks).strip()
        if not combined:
            raise RuntimeError(f"{self.name} response content was empty.")
        return combined

    def generate(self, *, prompt: str, task: Task, k: int, seed: int) -> list[str]:
        _ = (task, seed)
        outputs: list[str] = []
        for _idx in range(k):
            outputs.append(self._request_completion(prompt))
        return outputs


@dataclass
class GeminiModel:
    name: str = "gemini"
    model_name: str = ""
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 1024
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name:
            self.model_name = os.getenv("TCT_GEMINI_MODEL", "")

    def _request_completion(self, prompt: str) -> str:
        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for model `{self.name}`.")
        if not self.model_name:
            raise RuntimeError(
                f"No model_name configured for `{self.name}`. Set TCT_GEMINI_MODEL or model config."
            )

        model_id = self.model_name.strip()
        if model_id.startswith("models/"):
            model_id = model_id[len("models/") :]
        encoded_key = urllib.parse.quote(api_key, safe="")
        url = f"{self.api_base.rstrip('/')}/models/{model_id}:generateContent?key={encoded_key}"
        payload = {
            "systemInstruction": {
                "parts": [
                    {"text": "Return only Python code implementing the requested function."},
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        headers = {"content-type": "application/json"}
        headers.update(self.extra_headers)

        raw = _post_json_request(
            url=url,
            payload=payload,
            headers=headers,
            model_label=self.name,
            timeout_seconds=60,
        )
        candidates = raw.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(f"{self.name} response had no candidates.")
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list):
            raise RuntimeError(f"{self.name} response parts were invalid.")

        chunks: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                chunks.append(text)

        combined = "\n".join(chunks).strip()
        if not combined:
            raise RuntimeError(f"{self.name} response content was empty.")
        return combined

    def generate(self, *, prompt: str, task: Task, k: int, seed: int) -> list[str]:
        _ = (task, seed)
        outputs: list[str] = []
        for _idx in range(k):
            outputs.append(self._request_completion(prompt))
        return outputs


@dataclass
class AliasModel:
    name: str
    inner: ModelAdapter

    def generate(self, *, prompt: str, task: Task, k: int, seed: int) -> list[str]:
        return self.inner.generate(prompt=prompt, task=task, k=k, seed=seed)


def builtin_models() -> dict[str, ModelAdapter]:
    return {
        "heuristic-small": HeuristicSmallModel(),
        "heuristic-medium": HeuristicMediumModel(),
        "reference-oracle": ReferenceOracleModel(),
        "openai-compatible": OpenAICompatibleModel(),
        "anthropic": AnthropicModel(),
        "gemini": GeminiModel(),
    }


def _parse_model_config(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model config must be a JSON object.")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("Model config field `models` must be a list.")
    return [dict(spec) for spec in models]


def _is_valid_env_var_name(name: str) -> bool:
    if not name:
        return False
    first = name[0]
    if not (first.isalpha() or first == "_"):
        return False
    for char in name[1:]:
        if not (char.isalnum() or char == "_"):
            return False
    return True


def _read_float_field(spec: dict[str, Any], field: str, default: float, *, model_name: str) -> float:
    raw = spec.get(field, default)
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"Model `{model_name}` field `{field}` must be a number, got `{raw}`.") from exc


def _read_int_field(spec: dict[str, Any], field: str, default: int, *, model_name: str) -> int:
    raw = spec.get(field, default)
    try:
        value = int(raw)
    except Exception as exc:
        raise ValueError(f"Model `{model_name}` field `{field}` must be an integer, got `{raw}`.") from exc
    if value < 1:
        raise ValueError(f"Model `{model_name}` field `{field}` must be >= 1.")
    return value


def _validate_model_spec(spec: dict[str, Any]) -> None:
    name = str(spec.get("name", "")).strip()
    if not name:
        raise ValueError("Each model spec requires a non-empty `name`.")

    kind = str(spec.get("type", "openai-compatible")).strip().lower()
    supported = {"builtin", "openai-compatible", "anthropic", "gemini"}
    if kind not in supported:
        options = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported model type `{kind}` in config for `{name}`. Supported: {options}")

    if kind == "builtin":
        builtin_name = str(spec.get("builtin", name)).strip()
        if not builtin_name:
            raise ValueError(f"Model `{name}` type `builtin` requires non-empty field `builtin`.")
        return

    model_name = str(spec.get("model", "")).strip()
    if not model_name:
        raise ValueError(f"Model `{name}`: missing required field `model` for type `{kind}`.")

    _read_float_field(spec, "temperature", 0.2, model_name=name)
    _read_int_field(spec, "max_tokens", 512, model_name=name)

    api_key_env = str(
        spec.get(
            "api_key_env",
            "OPENAI_API_KEY" if kind == "openai-compatible" else "ANTHROPIC_API_KEY" if kind == "anthropic" else "GEMINI_API_KEY",
        )
    ).strip()
    if not api_key_env:
        raise ValueError(f"Model `{name}` field `api_key_env` must be non-empty.")
    if not _is_valid_env_var_name(api_key_env):
        raise ValueError(f"Model `{name}` has invalid api_key_env `{api_key_env}`.")
    if api_key_env not in os.environ:
        print(
            f"Warning: `{api_key_env}` not set for model `{name}`. "
            "It may still be injected at runtime.",
            file=sys.stderr,
        )

    headers = spec.get("headers", {})
    if headers is not None and not isinstance(headers, dict):
        raise ValueError(f"Model `{name}` field `headers` must be an object.")


def models_from_config(path: str | Path) -> dict[str, ModelAdapter]:
    registry = builtin_models()
    specs = _parse_model_config(path)

    for spec in specs:
        _validate_model_spec(spec)
        name = str(spec.get("name", "")).strip()
        kind = str(spec.get("type", "openai-compatible")).strip().lower()
        if kind == "builtin":
            builtin_name = str(spec.get("builtin", name)).strip()
            if builtin_name not in registry:
                supported = ", ".join(sorted(registry))
                raise ValueError(
                    f"Unknown builtin `{builtin_name}` in model config `{name}`. Supported: {supported}"
                )
            registry[name] = AliasModel(name=name, inner=registry[builtin_name])
            continue

        model_name = str(spec.get("model", "")).strip()
        temperature = _read_float_field(spec, "temperature", 0.2, model_name=name)
        max_tokens = _read_int_field(spec, "max_tokens", 512, model_name=name)
        headers = spec.get("headers", {})
        if headers is None:
            headers = {}
        header_map = {str(k): str(v) for k, v in headers.items()}

        if kind == "openai-compatible":
            api_base = str(spec.get("api_base", "https://api.openai.com/v1")).strip()
            api_key_env = str(spec.get("api_key_env", "OPENAI_API_KEY")).strip()
            registry[name] = OpenAICompatibleModel(
                name=name,
                model_name=model_name,
                api_base=api_base,
                api_key_env=api_key_env,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=header_map,
            )
            continue

        if kind == "anthropic":
            api_base = str(spec.get("api_base", "https://api.anthropic.com/v1")).strip()
            api_key_env = str(spec.get("api_key_env", "ANTHROPIC_API_KEY")).strip()
            anthropic_version = str(spec.get("anthropic_version", "2023-06-01")).strip()
            registry[name] = AnthropicModel(
                name=name,
                model_name=model_name,
                api_base=api_base,
                api_key_env=api_key_env,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_version=anthropic_version,
                extra_headers=header_map,
            )
            continue

        if kind == "gemini":
            api_base = str(
                spec.get("api_base", "https://generativelanguage.googleapis.com/v1beta")
            ).strip()
            api_key_env = str(spec.get("api_key_env", "GEMINI_API_KEY")).strip()
            registry[name] = GeminiModel(
                name=name,
                model_name=model_name,
                api_base=api_base,
                api_key_env=api_key_env,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=header_map,
            )
            continue

        raise ValueError(f"Unsupported model type `{kind}` in config for `{name}`.")

    return registry


def resolve_models(
    model_names: list[str],
    *,
    model_config_path: str | Path | None = None,
) -> list[ModelAdapter]:
    registry = models_from_config(model_config_path) if model_config_path else builtin_models()
    missing = [name for name in model_names if name not in registry]
    if missing:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unknown model(s): {', '.join(missing)}. Supported: {supported}")
    return [registry[name] for name in model_names]
