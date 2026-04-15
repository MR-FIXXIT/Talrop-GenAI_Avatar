from __future__ import annotations

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_PROMPTS_DIR = Path(__file__).parent


_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=StrictUndefined,
    autoescape=False,
)


def render_template(template_path: str, *, vars: dict) -> str:
    tpl = _env.get_template(template_path)
    return tpl.render(**vars).strip()


def contextualize_system_prompt() -> str:
    return render_template("contextualize/system.jinja", vars={})


def qa_fact_extract_system_prompt(*, context: str) -> str:
    return render_template("qa/fact_extract_system.jinja", vars={"context": context})


def qa_faithful_answer_system_prompt(*, context: str, supported_facts: str, tone: Optional[str] = None) -> str:
    """Single-pass prompt: generates a faithful answer without a separate revision step."""
    return render_template(
        "qa/faithful_answer_system.jinja",
        vars={
            "context": context,
            "supported_facts": supported_facts,
            "tone": tone,
        },
    )


def build_effective_fact_extract_system_prompt(
    *,
    org_system_prompt: Optional[str],
    context: str,
) -> str:
    org_system = (org_system_prompt or "").strip()
    base = qa_fact_extract_system_prompt(context=context)
    if not org_system:
        return base
    return (org_system + "\n\n" + base).strip()


def build_effective_faithful_answer_system_prompt(
    *,
    org_system_prompt: Optional[str],
    context: str,
    supported_facts: str,
    tone: Optional[str] = None,
) -> str:
    """Builds the system prompt for the single-pass generate+faithfulness step."""
    org_system = (org_system_prompt or "").strip()
    base = qa_faithful_answer_system_prompt(
        context=context,
        supported_facts=supported_facts,
        tone=tone,
    )
    if not org_system:
        return base
    return (org_system + "\n\n" + base).strip()