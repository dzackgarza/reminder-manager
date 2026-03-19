from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    path: str


def expand_home(path: str) -> str:
    if not path.startswith("~/"):
        return path
    return f"{Path.home()}/{path[2:]}"


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def default_skills_dirs() -> list[str]:
    import os

    env_dirs = [
        entry.strip() for entry in os.environ.get("REMINDER_INJECTION_SKILLS_DIRS", "").split(":")
    ]
    env_dirs = [entry for entry in env_dirs if entry]
    if env_dirs:
        return unique([expand_home(entry) for entry in env_dirs])
    return unique([expand_home("~/.config/opencode/skills")])


def parse_frontmatter_value(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*[\"']?(.+?)[\"']?$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_skill_record(path: str, text: str) -> SkillRecord | None:
    name = parse_frontmatter_value(text, "name")
    description = parse_frontmatter_value(text, "description")
    if not name or not description:
        return None
    return SkillRecord(name=name, description=description, path=path)


def load_skill_records(skills_dirs: list[str]) -> list[SkillRecord]:
    candidates: list[str] = []
    for directory in skills_dirs:
        base = Path(directory).expanduser()
        if not base.exists():
            continue
        candidates.extend(str(path) for path in base.rglob("SKILL.md"))

    records: list[SkillRecord] = []
    seen: set[str] = set()
    for path in unique(candidates):
        text = Path(path).read_text(encoding="utf-8")
        record = parse_skill_record(path, text)
        if record is None or record.name in seen:
            continue
        seen.add(record.name)
        records.append(record)
    return records


def tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 3]


def lexical_overlap_score(prompt: str, skill: SkillRecord) -> float:
    prompt_tokens = set(tokenize(prompt))
    if not prompt_tokens:
        return 0.0
    skill_tokens = set(tokenize(f"{skill.name} {skill.description}"))
    if not skill_tokens:
        return 0.0
    overlap = sum(1 for token in prompt_tokens if token in skill_tokens)
    return overlap / len(prompt_tokens)


def top_skills_for_prompt(
    prompt: str, top_k: int, skills_dirs: list[str] | None = None
) -> list[SkillRecord]:
    skills = load_skill_records(skills_dirs or default_skills_dirs())
    ranked = sorted(
        ((skill, lexical_overlap_score(prompt, skill)) for skill in skills),
        key=lambda entry: entry[1],
        reverse=True,
    )
    return [skill for skill, _score in ranked[:top_k] if _score > 0]


class SkillCache:
    def __init__(
        self,
        skills_dirs: list[str] | None = None,
        records: list[SkillRecord] | None = None,
    ) -> None:
        self.skills_dirs = skills_dirs or default_skills_dirs()
        self._records: list[SkillRecord] | None = records

    def list_skills(self) -> list[SkillRecord]:
        if self._records is None:
            self._records = load_skill_records(self.skills_dirs)
        return self._records

    def top_skills_for_prompt(self, prompt: str, top_k: int) -> list[SkillRecord]:
        skills = self.list_skills()
        ranked = sorted(
            ((skill, lexical_overlap_score(prompt, skill)) for skill in skills),
            key=lambda entry: entry[1],
            reverse=True,
        )
        return [skill for skill, score in ranked[:top_k] if score > 0]
