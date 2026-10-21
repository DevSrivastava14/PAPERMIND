"""Targeted unit tests for the retrieval evaluation dataset infrastructure."""

from pathlib import Path
import tempfile
import pytest

from app.evaluation import (
    EvaluationDataset,
    EvaluationQuestion,
    GoldEvidence,
    QuestionType,
    get_retrieval_evaluation_dataset,
    get_retrieval_evaluation_questions,
)


def test_evaluation_dataset_contains_exact_8_questions():
    """Verify that the dataset contains exactly 8 questions with IDs Q1 to Q8."""
    dataset = get_retrieval_evaluation_dataset()
    assert len(dataset) == 8

    expected_ids = [f"Q{i}" for i in range(1, 9)]
    actual_ids = [q.id for q in dataset]
    assert actual_ids == expected_ids


def test_question_types_are_correct():
    """Verify each question has its expected categorical question type."""
    dataset = get_retrieval_evaluation_dataset()

    assert dataset["Q1"].question_type == QuestionType.CONCEPTUAL
    assert dataset["Q2"].question_type == QuestionType.CONCEPTUAL
    assert dataset["Q3"].question_type == QuestionType.KEYWORD_HEAVY
    assert dataset["Q4"].question_type == QuestionType.KEYWORD_HEAVY
    assert dataset["Q5"].question_type == QuestionType.ARCHITECTURE
    assert dataset["Q6"].question_type == QuestionType.FIGURE_CAPTION
    assert dataset["Q7"].question_type == QuestionType.SPECIFIC_FACTUAL
    assert dataset["Q8"].question_type == QuestionType.SPECIFIC_FACTUAL


def test_q1_and_q4_have_multiple_gold_evidence_chunks():
    """Verify the multi-chunk distinction for Q1 and Q4 to support partial and complete retrieval evaluation."""
    dataset = get_retrieval_evaluation_dataset()

    # Q1: 2 gold evidence chunks on different pages (p. 2 and p. 3)
    q1 = dataset["Q1"]
    assert q1.has_multiple_evidence is True
    assert q1.is_multi_chunk is True
    assert len(q1.primary_evidence) == 2
    assert q1.gold_pages == [2, 3]
    assert "tests multi-chunk retrieval" in (q1.special_notes or "")

    # Q4: 3 gold evidence chunks for 3 distinct attacks on p. 4
    q4 = dataset["Q4"]
    assert q4.has_multiple_evidence is True
    assert q4.is_multi_chunk is True
    assert len(q4.primary_evidence) == 3
    assert q4.gold_pages == [4]
    assert any("data poisoning" in str(e.metadata) for e in q4.primary_evidence)
    assert any("model poisoning" in str(e.metadata) for e in q4.primary_evidence)
    assert any("model inversion" in str(e.metadata) for e in q4.primary_evidence)

    # All multi-evidence questions in dataset must be exactly Q1 and Q4
    multi_evidence_ids = [q.id for q in dataset.multi_evidence_questions]
    assert multi_evidence_ids == ["Q1", "Q4"]


def test_single_evidence_questions():
    """Verify that questions other than Q1 and Q4 are single-evidence chunks."""
    dataset = get_retrieval_evaluation_dataset()

    for qid in ["Q2", "Q3", "Q5", "Q6", "Q7", "Q8"]:
        q = dataset[qid]
        assert q.has_multiple_evidence is False, f"Expected {qid} to have single primary evidence"
        assert len(q.primary_evidence) == 1, f"Expected {qid} to have 1 primary evidence chunk"


def test_q6_figure_caption_special_note():
    """Verify that Q6 preserves the separate caption and visual breakdown evaluation note."""
    dataset = get_retrieval_evaluation_dataset()
    q6 = dataset["Q6"]

    assert q6.question_type == QuestionType.FIGURE_CAPTION
    assert q6.requires_separate_caption_scoring is True
    assert q6.special_notes is not None
    assert "found the caption" in q6.special_notes
    assert "answered the sub-type detail" in q6.special_notes

    figure_questions = dataset.figure_caption_questions
    assert len(figure_questions) == 1
    assert figure_questions[0].id == "Q6"


def test_q3_distractor_and_special_note():
    """Verify that Q3 preserves reference [22] distractor check and special note."""
    dataset = get_retrieval_evaluation_dataset()
    q3 = dataset["Q3"]

    assert q3.special_notes is not None
    assert "distractor check" in q3.special_notes
    assert len(q3.primary_evidence) == 1
    assert q3.primary_evidence[0].page_number == 4

    distractors = [e for e in q3.gold_evidence if e.is_distractor]
    assert len(distractors) == 1
    assert distractors[0].page_number == 6


def test_q5_nist_ai_rmf_special_note():
    """Verify Q5 architecture query preserves chunking note regarding intro and 4 bullets."""
    dataset = get_retrieval_evaluation_dataset()
    q5 = dataset["Q5"]

    assert q5.question_type == QuestionType.ARCHITECTURE
    assert q5.special_notes is not None
    assert "keeps the intro and all four bullets together" in q5.special_notes
    assert q5.gold_evidence[0].page_number == 5


def test_queries_and_reference_answers_verbatim_match_markdown():
    """Verify all queries and reference answers match data/evaluation/retrieval_questions.md verbatim."""
    md_path = Path("data/evaluation/retrieval_questions.md")
    assert md_path.exists(), "retrieval_questions.md must exist"

    md_content = md_path.read_text(encoding="utf-8")
    dataset = get_retrieval_evaluation_dataset()

    for q in dataset:
        # Check query appears verbatim in the markdown file
        assert q.query in md_content, f"Query for {q.id} not found verbatim in markdown: '{q.query}'"
        # Check reference answer content is present in markdown
        first_line_ref = q.reference_answer.split("\n")[0].strip()
        assert first_line_ref in md_content, (
            f"Reference answer for {q.id} not found verbatim in markdown: '{first_line_ref}'"
        )


def test_dataset_indexing_and_filtering():
    """Verify dictionary-style access, case-insensitive ID retrieval, and type filtering."""
    dataset = get_retrieval_evaluation_dataset()

    # Indexing by int
    assert dataset[0].id == "Q1"

    # Indexing by str
    assert dataset["Q2"].id == "Q2"
    assert dataset.get("q3") is not None
    assert dataset.get("q3").id == "Q3"
    assert dataset.get("NONEXISTENT") is None

    # Filtering by QuestionType
    conceptual = dataset.filter_by_type(QuestionType.CONCEPTUAL)
    assert [q.id for q in conceptual] == ["Q1", "Q2"]

    factual = dataset.filter_by_type("specific factual")
    assert [q.id for q in factual] == ["Q7", "Q8"]


def test_gold_evidence_validation():
    """Verify validation on GoldEvidence fields."""
    with pytest.raises(ValueError, match="text cannot be empty"):
        GoldEvidence(text="")

    with pytest.raises(ValueError, match="page_number must be an integer >= 1"):
        GoldEvidence(text="valid text", page_number=0)

    evidence = GoldEvidence(text="valid text", element_id="test#chunk_1", page_number=2, section="II.A")
    ev_dict = evidence.to_dict()
    assert ev_dict["text"] == "valid text"
    assert ev_dict["element_id"] == "test#chunk_1"
    assert ev_dict["page_number"] == 2

    restored = GoldEvidence.from_dict(ev_dict)
    assert restored.text == evidence.text
    assert restored.element_id == evidence.element_id
    assert restored.page_number == evidence.page_number


def test_evaluation_question_validation():
    """Verify validation on EvaluationQuestion fields."""
    ev = GoldEvidence(text="Evidence snippet", page_number=1)

    with pytest.raises(ValueError, match="Question id cannot be empty"):
        EvaluationQuestion(id="", query="query", reference_answer="answer", gold_evidence=[ev])

    with pytest.raises(ValueError, match="Question query cannot be empty"):
        EvaluationQuestion(id="Q1", query="  ", reference_answer="answer", gold_evidence=[ev])

    with pytest.raises(ValueError, match="Reference answer cannot be empty"):
        EvaluationQuestion(id="Q1", query="query", reference_answer="", gold_evidence=[ev])

    with pytest.raises(ValueError, match="must have at least one gold evidence chunk"):
        EvaluationQuestion(id="Q1", query="query", reference_answer="answer", gold_evidence=[])


def test_dataset_json_serialization_round_trip():
    """Verify serialization to/from JSON."""
    dataset = get_retrieval_evaluation_dataset()
    as_dict = dataset.to_dict()

    restored_from_dict = EvaluationDataset.from_dict(as_dict)
    assert len(restored_from_dict) == len(dataset)
    assert restored_from_dict["Q1"].query == dataset["Q1"].query

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test_eval_dataset.json"
        dataset.to_json(tmp_file)
        assert tmp_file.exists()

        restored_from_file = EvaluationDataset.from_json(tmp_file)
        assert len(restored_from_file) == 8
        assert restored_from_file["Q6"].special_notes == dataset["Q6"].special_notes


def test_deterministic_gold_evidence_element_id_mappings():
    """Verify that every gold evidence item maps to an actual existing element_id in sample_paper.pdf."""
    from app.evaluation import build_sample_paper_document

    doc = build_sample_paper_document("data/sample_paper.pdf")
    dataset = get_retrieval_evaluation_dataset()

    for q in dataset:
        for ev in q.gold_evidence:
            assert ev.element_id is not None, f"{q.id} evidence element_id is None"
            # Must exist in actual document (no invented IDs)
            assert ev.element_id in doc, f"Element ID '{ev.element_id}' for {q.id} not found in sample_paper.pdf"
            el = doc[ev.element_id]
            assert el is not None
            assert el.page_number == ev.page_number

    # Q1: multiple gold element IDs across pages 2 and 3
    q1 = dataset["Q1"]
    assert q1.gold_element_ids == ["sample_paper.pdf#chunk_10", "sample_paper.pdf#chunk_14"]
    assert len(q1.unique_gold_element_ids) == 2

    # Q2: maps to chunk_10 on page 2
    q2 = dataset["Q2"]
    assert q2.gold_element_ids == ["sample_paper.pdf#chunk_10"]

    # Q3: primary is chunk_23; distractor is chunk_41 and must NOT be in gold_element_ids
    q3 = dataset["Q3"]
    assert q3.gold_element_ids == ["sample_paper.pdf#chunk_23"]
    assert "sample_paper.pdf#chunk_41" not in q3.gold_element_ids
    distractor = [e for e in q3.gold_evidence if e.is_distractor][0]
    assert distractor.element_id == "sample_paper.pdf#chunk_41"

    # Q4: multiple gold element IDs across chunks 21 and 22 on page 4
    q4 = dataset["Q4"]
    assert set(q4.unique_gold_element_ids) == {"sample_paper.pdf#chunk_21", "sample_paper.pdf#chunk_22"}
    assert len(q4.gold_element_ids) == 3

    # Q5: maps to chunk_31 containing all 4 functions
    q5 = dataset["Q5"]
    assert q5.gold_element_ids == ["sample_paper.pdf#chunk_31"]

    # Q6: maps to Figure 2 caption element_id
    q6 = dataset["Q6"]
    assert q6.gold_element_ids == ["sample_paper.pdf#page_2_cap_1"]
    cap_el = doc["sample_paper.pdf#page_2_cap_1"]
    assert cap_el.element_type.value == "caption"
    assert q6.requires_separate_caption_scoring is True

    # Q7: maps to chunk_18 (Amazon applicant scandal 2018)
    q7 = dataset["Q7"]
    assert q7.gold_element_ids == ["sample_paper.pdf#chunk_18"]

    # Q8: maps to chunk_14 (Qrator Labs DDoS detection model)
    q8 = dataset["Q8"]
    assert q8.gold_element_ids == ["sample_paper.pdf#chunk_14"]

