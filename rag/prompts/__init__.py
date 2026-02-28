from __future__ import annotations

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_PROMPTS_DIR = Path(__file__).parent


_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=StrictUndefined,  # fails fast if you forget to pass a variable like context
    autoescape=False,
)


def render_template(template_path: str, *, vars: dict) -> str:
    tpl = _env.get_template(template_path)
    return tpl.render(**vars).strip()


def qa_system_prompt(*, context: str) -> str:
    return render_template("qa/system.jinja", vars={"context": context})


def contextualize_system_prompt() -> str:
    # no variables needed
    return render_template("contextualize/system.jinja", vars={})


def build_effective_qa_system_prompt(*, org_system_prompt: Optional[str], context: str) -> str:
    org_system = (org_system_prompt or "").strip()
    base = qa_system_prompt(context=context)
    if not org_system:
        return base
    return (org_system + "\n\n" + base).strip()