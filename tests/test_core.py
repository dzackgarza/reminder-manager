from skill_suggester.core import SkillCache, SkillRecord, lexical_overlap_score, parse_skill_record


def test_parse_skill_record_reads_name_and_description() -> None:
    text = "name: foo\ndescription: bar\n"
    record = parse_skill_record("/tmp/SKILL.md", text)
    assert record is not None
    assert record.name == "foo"
    assert record.description == "bar"


def test_lexical_overlap_score_is_positive_for_matching_terms() -> None:
    score = lexical_overlap_score(
        "use zotero export tool",
        SkillRecord(name="zotero", description="export library metadata", path="/tmp/skill"),
    )
    assert score > 0


def test_skill_cache_uses_loaded_records() -> None:
    cache = SkillCache(
        skills_dirs=[],
        records=[
            SkillRecord(name="zotero", description="export library metadata", path="/tmp/skill")
        ],
    )
    matches = cache.top_skills_for_prompt("need export metadata", 3)
    assert matches[0].name == "zotero"
