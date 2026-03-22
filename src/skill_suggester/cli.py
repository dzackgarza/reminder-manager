from __future__ import annotations

import json
import os
from pathlib import Path

from cyclopts import App
from pydantic import BaseModel, ConfigDict, Field, validate_call

from .core import DEFAULT_MODEL, SkillCache, default_skills_dirs

app = App(name="skill-suggester", help="Suggest relevant skills for a prompt.")


class DoctorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills_dirs: list[str]
    skills_dir_count: int
    existing_skills_dirs: list[str]
    discovered_skill_count: int
    model_name: str
    transformers_backend_enabled: bool
    environment_ok: bool


@app.command
@validate_call
def doctor() -> None:
    skills_dirs = default_skills_dirs()
    cache = SkillCache(skills_dirs)
    existing_dirs = [path for path in skills_dirs if Path(path).expanduser().exists()]
    model_name = os.environ.get("REMINDER_INJECTION_MODEL", DEFAULT_MODEL).strip()
    report = DoctorReport(
        skills_dirs=skills_dirs,
        skills_dir_count=len(skills_dirs),
        existing_skills_dirs=existing_dirs,
        discovered_skill_count=len(cache.list_skills()),
        model_name=model_name,
        transformers_backend_enabled=True,
        environment_ok=bool(existing_dirs),
    )
    print(report.model_dump_json())


@app.command(name="top-skills")
@validate_call
def top_skills(prompt: str, top_k: int = Field(default=3, ge=1)) -> None:
    skills = SkillCache().top_skills_for_prompt(prompt, top_k)
    print(json.dumps([skill.model_dump(exclude={"embedding"}) for skill in skills]))


def main() -> None:
    app()
