from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict

type Extractor = Callable[[list[str]], list[list[float]]]

DEFAULT_MODEL = os.environ.get(
    "REMINDER_INJECTION_MODEL",
    "mixedbread-ai/mxbai-embed-xsmall-v1",
).strip()

_shared_extractors: dict[str, Extractor] = {}


class SkillRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    path: str
    embedding: list[float] | None = None


class SupportsEncode(Protocol):
    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Sequence[Sequence[float]]:
        ...


def expand_home(path: str) -> str:
    if not path.startswith("~/"):
        return path
    return f"{Path.home()}/{path[2:]}"


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def default_skills_dirs() -> list[str]:
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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left)
    right_norm = sum(value * value for value in right)
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return dot / ((left_norm**0.5) * (right_norm**0.5))


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


def build_extractor(model_name: str) -> Extractor:
    from sentence_transformers import SentenceTransformer

    model = cast(SupportsEncode, SentenceTransformer(model_name))

    def extract(texts: list[str]) -> list[list[float]]:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in embeddings]

    return extract


def get_shared_extractor(model_name: str = DEFAULT_MODEL) -> Extractor:
    extractor = _shared_extractors.get(model_name)
    if extractor is None:
        extractor = build_extractor(model_name)
        _shared_extractors[model_name] = extractor
    return extractor


def embed_skill_records(records: list[SkillRecord], extractor: Extractor) -> list[SkillRecord]:
    if not records:
        return []
    embeddings = extractor([f"{record.name}: {record.description}" for record in records])
    return [
        record.model_copy(update={"embedding": [float(value) for value in embeddings[index]]})
        for index, record in enumerate(records)
    ]


def top_skills_for_prompt(
    prompt: str,
    top_k: int,
    skills_dirs: list[str] | None = None,
    extractor: Extractor | None = None,
) -> list[SkillRecord]:
    cache = SkillCache(
        skills_dirs=skills_dirs or default_skills_dirs(),
        extractor=extractor,
    )
    return cache.top_skills_for_prompt(prompt, top_k)


class SkillCache:
    def __init__(
        self,
        skills_dirs: list[str] | None = None,
        extractor: Extractor | None = None,
        records: list[SkillRecord] | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.skills_dirs = skills_dirs or default_skills_dirs()
        self.model = model
        self.extractor = extractor or get_shared_extractor(model)
        self._records = records
        self._embedded_records: list[SkillRecord] | None = None

    def list_skills(self) -> list[SkillRecord]:
        if self._records is None:
            self._records = load_skill_records(self.skills_dirs)
        return self._records

    def top_skills_for_prompt(self, prompt: str, top_k: int) -> list[SkillRecord]:
        skills = self._embedded_skill_records()
        if not skills:
            return []
        prompt_embedding = self.extractor([prompt])[0]
        ranked = sorted(
            (
                (
                    skill,
                    cosine_similarity(prompt_embedding, skill.embedding or [])
                    + lexical_overlap_score(prompt, skill),
                )
                for skill in skills
            ),
            key=lambda entry: entry[1],
            reverse=True,
        )
        return [skill for skill, _score in ranked[:top_k]]

    def _embedded_skill_records(self) -> list[SkillRecord]:
        if self._embedded_records is None:
            self._embedded_records = embed_skill_records(self.list_skills(), self.extractor)
        return self._embedded_records
