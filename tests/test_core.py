from __future__ import annotations

from pathlib import Path

import pytest

from skill_suggester.core import (
    SkillCache,
    SkillRecord,
    cosine_similarity,
    default_skills_dirs,
    lexical_overlap_score,
    load_skill_records,
    parse_skill_record,
)


def write_skill(root: Path, name: str, description: str, body: str | None = None) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f'description: "{description}"',
                "---",
                "",
                body or f"# {name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def fake_extractor(texts: list[str]) -> list[list[float]]:
    def vector_for(text: str) -> list[float]:
        lowered = text.lower()
        axes = {
            "justfile": ["justfile", "build", "automation", "recipe"],
            "ast-grep": ["ast", "structural", "import", "function"],
            "zotero": ["zotero", "doi", "bibliography", "citation", "reference"],
            "latex": ["latex", "compile", "bibliography", "citation"],
            "opencode-cli": ["opencode", "tool", "session", "agent", "runtime"],
        }
        return [
            1.0 if any(token in lowered for token in tokens) else 0.0
            for tokens in axes.values()
        ]

    return [vector_for(text) for text in texts]


def test_parse_skill_record_reads_name_and_description() -> None:
    record = parse_skill_record("/tmp/SKILL.md", "name: foo\ndescription: bar\n")

    assert record is not None
    assert record.name == "foo"
    assert record.description == "bar"


def test_default_skills_dirs_prefers_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REMINDER_INJECTION_SKILLS_DIRS", "~/one:~/two")

    assert default_skills_dirs() == [f"{Path.home()}/one", f"{Path.home()}/two"]


def test_load_skill_records_deduplicates_names_and_ignores_invalid_files(tmp_path: Path) -> None:
    canonical = tmp_path / "skills"
    duplicate = tmp_path / "duplicate"
    invalid = tmp_path / "invalid"
    write_skill(canonical, "justfile", "Build automation and task runner recipes.")
    write_skill(canonical, "zotero", "Bibliography metadata and DOI cleanup.")
    write_skill(duplicate, "justfile", "Duplicate skill that should be ignored.")
    broken = invalid / "broken"
    broken.mkdir(parents=True, exist_ok=True)
    broken.joinpath("SKILL.md").write_text("# missing frontmatter\n", encoding="utf-8")

    records = load_skill_records([str(canonical), str(duplicate), str(invalid)])

    assert sorted(record.name for record in records) == ["justfile", "zotero"]


def test_cosine_similarity_matches_aligned_opposed_and_mismatched_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == -1.0


def test_skill_cache_ranks_semantically_relevant_skills_first(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    write_skill(
        skills_dir,
        "justfile",
        "Use when working with just recipes, automation, and task runner maintenance.",
    )
    write_skill(
        skills_dir,
        "ast-grep",
        "Use when searching code structure with AST patterns and import matching.",
    )
    write_skill(
        skills_dir,
        "zotero",
        "Use when fixing DOI metadata, bibliography exports, and reference cleanup.",
    )
    write_skill(
        skills_dir,
        "latex-compile-qa",
        "Use when LaTeX citation and bibliography compilation is broken.",
    )
    write_skill(
        skills_dir,
        "opencode-cli",
        "Use when listing tools, sessions, agents, and OpenCode runtime state.",
    )

    cache = SkillCache(skills_dirs=[str(skills_dir)], extractor=fake_extractor)

    cases = [
        (
            "Update the build automation and the just recipes for this project.",
            "justfile",
        ),
        (
            (
                "Search for import declarations and function definitions with a "
                "structural AST matcher."
            ),
            "ast-grep",
        ),
        (
            "Fix the DOI metadata and clean up the bibliography citations in my reference library.",
            "zotero",
        ),
        (
            "The LaTeX paper does not compile because references and citations are broken.",
            "latex-compile-qa",
        ),
        (
            "List all tool names available in this OpenCode session.",
            "opencode-cli",
        ),
    ]

    for prompt, expected_first in cases:
        top = cache.top_skills_for_prompt(prompt, 3)
        assert top[0].name == expected_first
        assert len(top) == 3


def test_lexical_overlap_breaks_embedding_ties_for_related_terms() -> None:
    prompt = "need bibliography citation cleanup"
    zotero = SkillRecord(
        name="zotero",
        description="reference cleanup and DOI metadata export",
        path="/tmp/zotero",
        embedding=[0.0, 0.0, 1.0],
    )
    latex = SkillRecord(
        name="latex-compile-qa",
        description="compile failures for LaTeX documents",
        path="/tmp/latex",
        embedding=[0.0, 0.0, 1.0],
    )
    cache = SkillCache(skills_dirs=[], records=[zotero, latex], extractor=fake_extractor)

    top = cache.top_skills_for_prompt(prompt, 2)

    assert lexical_overlap_score(prompt, zotero) > lexical_overlap_score(prompt, latex)
    assert [skill.name for skill in top] == ["zotero", "latex-compile-qa"]
