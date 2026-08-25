"""Optional Ragas adapter with in-process and isolated-process execution."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Literal, Optional


SUPPORTED_RAGAS_METRICS = {
    "id_context_precision",
    "id_context_recall",
    "context_precision",
    "context_recall",
}


@dataclass(frozen=True)
class RagasEvaluationSample:
    question: str
    retrieved_context_ids: list[str]
    reference_context_ids: list[str]
    retrieved_contexts: list[str]
    reference_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagasEvaluator:
    """Run selected Ragas metrics without making Ragas a core dependency."""

    def __init__(
        self,
        *,
        execution_mode: Literal["process", "in_process"] = "process",
        python_executable: Optional[str] = None,
        metrics: Optional[Iterable[str]] = None,
        timeout_seconds: float = 300.0,
        llm_model: str = "",
        llm_api_key: str = "",
        llm_base_url: str = "",
    ) -> None:
        selected = list(metrics or ("id_context_precision", "id_context_recall"))
        unknown = sorted(set(selected) - SUPPORTED_RAGAS_METRICS)
        if unknown:
            raise ValueError(f"Unsupported Ragas metrics: {', '.join(unknown)}")
        if not selected:
            raise ValueError("At least one Ragas metric must be configured")
        if timeout_seconds <= 0:
            raise ValueError("Ragas timeout must be positive")

        self.execution_mode = execution_mode
        self.python_executable = python_executable or sys.executable
        self.metrics = selected
        self.timeout_seconds = timeout_seconds
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url

    def evaluate(self, samples: Iterable[RagasEvaluationSample]) -> dict[str, Any]:
        payload = {
            "metrics": self.metrics,
            "samples": [sample.to_dict() for sample in samples],
            "llm": {
                "model": self.llm_model,
                "api_key": self.llm_api_key,
                "base_url": self.llm_base_url,
            },
        }
        try:
            if self.execution_mode == "in_process":
                result = self._evaluate_in_process(payload)
            else:
                result = self._evaluate_in_subprocess(payload)
        except FileNotFoundError as exc:
            result = {
                "status": "unavailable",
                "error": f"Ragas Python executable not found: {exc.filename}",
            }
        except subprocess.TimeoutExpired:
            result = {
                "status": "failed",
                "error": f"Ragas evaluation timed out after {self.timeout_seconds:g}s",
            }
        except Exception as exc:
            result = {"status": "failed", "error": str(exc)[:1000]}

        result.setdefault("status", "completed")
        result.setdefault("metrics", {})
        result["execution_mode"] = self.execution_mode
        result["configured_metrics"] = list(self.metrics)
        return result

    def _evaluate_in_process(self, payload: dict[str, Any]) -> dict[str, Any]:
        from backend.services.ragas_worker import evaluate_payload

        return asyncio.run(evaluate_payload(payload))

    def _evaluate_in_subprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        workspace_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [
                self.python_executable,
                "-m",
                "backend.services.ragas_worker",
            ],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=self.timeout_seconds,
            cwd=str(workspace_root),
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"Ragas worker exited with code {completed.returncode}: "
                f"{message[-1000:]}"
            )

        for line in reversed((completed.stdout or "").splitlines()):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(result, dict):
                return result
        raise RuntimeError("Ragas worker did not return a JSON result")


def _discover_ragas_python() -> Optional[str]:
    """Locate a dedicated Ragas virtual environment.

    Search order (Windows first, POSIX second): project-local venvs named
    ``.venv-ragas`` / ``venv-ragas`` / ``ragas-env``, then fall back to the
    current interpreter (the worker reports ``unavailable`` if Ragas is absent).
    An explicit ``RAGAS_PYTHON_EXECUTABLE`` always wins over auto-detection.
    """
    root = Path(__file__).resolve().parents[2]
    if sys.platform == "win32":
        candidates = [
            root / ".venv-ragas" / "Scripts" / "python.exe",
            root / "venv-ragas" / "Scripts" / "python.exe",
            root / "ragas-env" / "Scripts" / "python.exe",
            root / ".venv-ragas" / "bin" / "python",
        ]
    else:
        candidates = [
            root / ".venv-ragas" / "bin" / "python",
            root / "venv-ragas" / "bin" / "python",
            root / "ragas-env" / "bin" / "python",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def get_ragas_evaluator(settings, metrics=None) -> RagasEvaluator:
    # Per-run metric overrides take precedence over the global
    # RAGAS_METRICS setting, which keeps experiments reproducible.
    selected = metrics if metrics else [
        value.strip()
        for value in settings.RAGAS_METRICS.split(",")
        if value.strip()
    ]
    explicit = (settings.RAGAS_PYTHON_EXECUTABLE or "").strip()
    return RagasEvaluator(
        execution_mode=settings.RAGAS_EXECUTION_MODE,
        python_executable=explicit or _discover_ragas_python(),
        metrics=selected,
        timeout_seconds=settings.RAGAS_TIMEOUT,
        llm_model=settings.RAGAS_LLM_MODEL or settings.LLM_MODEL,
        llm_api_key=settings.RAGAS_LLM_API_KEY or settings.LLM_API_KEY,
        llm_base_url=settings.RAGAS_LLM_BASE_URL or settings.LLM_BASE_URL,
    )
