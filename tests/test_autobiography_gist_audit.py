from scripts.audit_autobiography_gists import audit_gist


def test_gist_audit_counts_cited_and_uncited_factual_sentences() -> None:
    report = audit_gist(
        session_id="s1",
        gist=(
            "I wrote the report [ep:e1]. "
            "I edited workspace/writing/4246/ch1.md [ep:e2]. "
            "I verified the health endpoint [ep:e3]. "
            "I created a proposal artifact. "
            "I felt curiosity while doing it."
        ),
        evidence_episode_ids=["e1", "e2", "e3"],
        existing_episode_ids={"e1", "e2", "e3"},
    )

    assert report.total_sentences == 5
    assert report.cited_sentences == 3
    assert report.uncited_factual_claims == 2
    assert report.mismatched_citations == 0


def test_gist_audit_reports_nonexistent_episode_citation() -> None:
    report = audit_gist(
        session_id="s1",
        gist="I wrote the report [ep:missing].",
        evidence_episode_ids=["e1"],
        existing_episode_ids={"e1"},
    )

    assert report.total_sentences == 1
    assert report.cited_sentences == 1
    assert report.uncited_factual_claims == 0
    assert report.mismatched_citations == 1
