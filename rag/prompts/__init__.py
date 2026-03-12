# from __future__ import annotations

# from pathlib import Path
# from typing import Optional

# from jinja2 import Environment, FileSystemLoader, StrictUndefined


# _PROMPTS_DIR = Path(__file__).parent


# _env = Environment(
#     loader=FileSystemLoader(str(_PROMPTS_DIR)),
#     undefined=StrictUndefined,  # fails fast if you forget to pass a variable like context
#     autoescape=False,
# )


# def render_template(template_path: str, *, vars: dict) -> str:
#     tpl = _env.get_template(template_path)
#     return tpl.render(**vars).strip()


# def qa_system_prompt(*, context: str) -> str:
#     return render_template("qa/system.jinja", vars={"context": context})


# def contextualize_system_prompt() -> str:
#     # no variables needed
#     return render_template("contextualize/system.jinja", vars={})


# def build_effective_qa_system_prompt(*, org_system_prompt: Optional[str], context: str) -> str:
#     org_system = (org_system_prompt or "").strip()
#     base = qa_system_prompt(context=context)
#     if not org_system:
#         return base
#     return (org_system + "\n\n" + base).strip()


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


def qa_answer_from_facts_system_prompt(*, context: str, supported_facts: str) -> str:
    return render_template(
        "qa/answer_from_facts_system.jinja",
        vars={
            "context": context,
            "supported_facts": supported_facts,
        },
    )


def qa_faithfulness_revision_system_prompt(*, context: str, draft_answer: str) -> str:
    return render_template(
        "qa/faithfulness_revision_system.jinja",
        vars={
            "context": context,
            "draft_answer": draft_answer,
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


def build_effective_answer_from_facts_system_prompt(
    *,
    org_system_prompt: Optional[str],
    context: str,
    supported_facts: str,
) -> str:
    org_system = (org_system_prompt or "").strip()
    base = qa_answer_from_facts_system_prompt(
        context=context,
        supported_facts=supported_facts,
    )
    if not org_system:
        return base
    return (org_system + "\n\n" + base).strip()


def build_effective_faithfulness_revision_system_prompt(
    *,
    org_system_prompt: Optional[str],
    context: str,
    draft_answer: str,
) -> str:
    org_system = (org_system_prompt or "").strip()
    base = qa_faithfulness_revision_system_prompt(
        context=context,
        draft_answer=draft_answer,
    )
    if not org_system:
        return base
    return (org_system + "\n\n" + base).strip()