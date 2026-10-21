"""Canonical retrieval evaluation dataset curated from data/evaluation/retrieval_questions.md.

This module provides the 8 curated evaluation questions with exact query wording,
reference answers, gold evidence chunks, deterministic element_id mappings to
data/sample_paper.pdf, question types, and evaluation notes.
"""

from __future__ import annotations

from typing import List

from app.evaluation.models import (
    EvaluationDataset,
    EvaluationQuestion,
    GoldEvidence,
    QuestionType,
)


def get_retrieval_evaluation_questions() -> List[EvaluationQuestion]:
    """Returns the 8 manually curated retrieval evaluation questions.

    Exact wording and notes are preserved from data/evaluation/retrieval_questions.md:
      - Q1 and Q4 have multiple gold evidence chunks with distinct element_ids
        to evaluate both partial (>= 1 gold chunk) and complete (all gold chunks) retrieval.
      - Q6 preserves the Figure 2 caption element_id ('sample_paper.pdf#page_2_cap_1')
        and its separate visual breakdown note.
      - Q3 preserves the reference [22] distractor chunk ('sample_paper.pdf#chunk_41')
        which is explicitly marked with is_distractor=True (NOT a positive gold element).
      - Q5 preserves the NIST AI RMF intro and 4 bullets note.
    """
    return [
        EvaluationQuestion(
            id="Q1",
            query='What is the difference between “AI for Security” and “Security for AI”?',
            reference_answer=(
                '“AI for Security” is using AI technologies to solve security problems, '
                'or building AI into security products. “Security for AI” is designing and '
                'implementing security measures to protect AI technologies and infrastructure, '
                'which matters most when the AI is critical to business or nation-state infrastructure.'
            ),
            question_type=QuestionType.CONCEPTUAL,
            gold_evidence=[
                GoldEvidence(
                    text=(
                        '‘AI for Security’ means how can we use AI technologies to solve security '
                        'problems or use AI technology in security products or solutions.'
                    ),
                    element_id="sample_paper.pdf#chunk_10",
                    page_number=2,
                    section="II.B",
                    metadata={"topic": "AI for Security definition"},
                ),
                GoldEvidence(
                    text=(
                        '‘Security for AI’ means designing, implementing security measures for AI technologies, '
                        'since every technology has limitations, vulnerabilities and prone to attacks so it is '
                        'imperative to secure AI infrastructure especially if its crucial to the business '
                        'and nation state infrastructure.'
                    ),
                    element_id="sample_paper.pdf#chunk_14",
                    page_number=3,
                    section="II.C",
                    metadata={"topic": "Security for AI definition"},
                ),
            ],
            special_notes="The two evidence chunks sit on different pages, so this tests multi-chunk retrieval.",
        ),
        EvaluationQuestion(
            id="Q2",
            query="Why does the author say Strong AI is not yet practical, and how does it differ from Weak AI?",
            reference_answer=(
                "Weak (Narrow) AI is trained for limited tasks and is already in use (Alexa, Siri). "
                "Strong AI would be artificial general intelligence that thinks like humans, "
                "with self-aware consciousness. It is still theoretical, with no real-world "
                "use cases, and researchers are only exploring it."
            ),
            question_type=QuestionType.CONCEPTUAL,
            gold_evidence=[
                GoldEvidence(
                    text=(
                        'Weak AI alternatively called Narrow AI, which has been trained to perform limited tasks. '
                        'Strong AI and its branches are theoretical concepts at the moment and not practical , '
                        'do not have real world use cases at the moment. Researchers are currently exploring this possibility [8].'
                    ),
                    element_id="sample_paper.pdf#chunk_10",
                    page_number=2,
                    section="II.A",
                    metadata={
                        "topic": "Weak AI vs Strong AI practicality",
                        "secondary_element_id": "sample_paper.pdf#chunk_9",
                    },
                ),
            ],
            special_notes=None,
        ),
        EvaluationQuestion(
            id="Q3",
            query="What is MITRE ATLAS and what is its purpose?",
            reference_answer=(
                "MITRE ATLAS (Adversarial Threat Landscape for Artificial-intelligence Systems) "
                "is a framework built by MITRE, a not-for-profit security organization, "
                "specifically for AI. It raises awareness of evolving vulnerabilities and "
                "of the tactics, techniques and procedures for tackling them."
            ),
            question_type=QuestionType.KEYWORD_HEAVY,
            gold_evidence=[
                GoldEvidence(
                    text=(
                        "MITRE, a prominent and not for profit security, has built a framework "
                        "specifically for AI called MITRE ATLAS (Adversarial Threat Landscape for "
                        "Artificial-intelligence Systems) to raise awareness of the evolving "
                        "vulnerabilities and tactics, techniques and procedures to tackle them [22]."
                    ),
                    element_id="sample_paper.pdf#chunk_23",
                    page_number=4,
                    section="end of II.C.3",
                    metadata={"topic": "MITRE ATLAS purpose and definition"},
                ),
                GoldEvidence(
                    text="reference [22] on p. 6",
                    element_id="sample_paper.pdf#chunk_41",
                    page_number=6,
                    section="References",
                    is_distractor=True,
                    metadata={"topic": "Citation reference distractor"},
                ),
            ],
            special_notes="Also relevant: reference [22] on p. 6 (a weak match, useful as a distractor check).",
        ),
        EvaluationQuestion(
            id="Q4",
            query="How do data poisoning, model poisoning, and model inversion attacks differ?",
            reference_answer=(
                "Data poisoning: the attacker spams data storage or inserts mislabeled or irrelevant data, "
                "so predictions become inaccurate and unreliable.\n"
                "Model poisoning: malicious data is introduced into the training data in a development or "
                "testing environment, which can shift the model’s behavior significantly. The paper contrasts "
                "this with adversarial attacks, which target models in production.\n"
                "Model inversion: the attacker reverse-engineers the model to expose the data set it used, "
                "giving access to PII/MNPI, and these attacks are hard to detect."
            ),
            question_type=QuestionType.KEYWORD_HEAVY,
            gold_evidence=[
                GoldEvidence(
                    text=(
                        "An attacker can intentionally spam data storage, insert incorrectly labeled data sets, "
                        "or irrelevant data sets deviating from the median range of the training set, "
                        "resulting in inaccurate predictions, insecure, and unreliable outcomes."
                    ),
                    element_id="sample_paper.pdf#chunk_21",
                    page_number=4,
                    section="II.C.3, item 2",
                    metadata={"attack_type": "data poisoning"},
                ),
                GoldEvidence(
                    text=(
                        "Adversarial attacks target AI models or systems in their production environment, "
                        "whereas model poisoning attacks target AI models in a development or testing environment."
                    ),
                    element_id="sample_paper.pdf#chunk_21",
                    page_number=4,
                    section="II.C.3, item 3",
                    metadata={
                        "attack_type": "model poisoning",
                        "continues_in": "sample_paper.pdf#chunk_22",
                    },
                ),
                GoldEvidence(
                    text=(
                        "An attacker can reverse engineer the model, thus exposing the data set being used, "
                        "resulting in access to PII/MNPI"
                    ),
                    element_id="sample_paper.pdf#chunk_22",
                    page_number=4,
                    section="II.C.3, item 4",
                    metadata={"attack_type": "model inversion"},
                ),
            ],
            special_notes="Three distinct attack definitions spanning items 2-4 in II.C.3; tests multi-chunk retrieval.",
        ),
        EvaluationQuestion(
            id="Q5",
            query="What are the four functions of the NIST AI RMF Core, and what does each one do?",
            reference_answer=(
                "GOVERN: ensures strong governance that drives internal practices and norms and builds an organizational risk culture.\n"
                "MAP: establishes the context to frame risks related to an AI system.\n"
                "MEASURE: uses quantitative, qualitative or mixed-method tools to analyze, assess, benchmark and monitor AI risks.\n"
                "MANAGE: allocates risk resources to mapped and measured risks on a regular basis, as defined by GOVERN."
            ),
            question_type=QuestionType.ARCHITECTURE,
            gold_evidence=[
                GoldEvidence(
                    text=(
                        "the four bullets beginning “GOVERN ensures strong governance...”, “The MAP function establishes "
                        "the context...”, “The MEASURE function employs...”, “The MANAGE function allocates risk resources...”"
                    ),
                    element_id="sample_paper.pdf#chunk_31",
                    page_number=5,
                    section="bullet list",
                    metadata={
                        "topic": "NIST AI RMF Core functions",
                        "intro_element_id": "sample_paper.pdf#chunk_30",
                    },
                ),
            ],
            special_notes=(
                "The list is introduced by “As seen in [27, Fig.3] there are following sections in the AI RMF Core:”, "
                "so a good chunker keeps the intro and all four bullets together."
            ),
        ),
        EvaluationQuestion(
            id="Q6",
            query="What does Figure 2 show, and how are Weak AI and Strong AI broken down?",
            reference_answer=(
                "Figure 2, captioned “Weak AI vs Strong AI,” is a tree diagram with AI at the top. "
                "It splits into Strong AI (Artificial General Intelligence and Artificial Super Intelligence) "
                "and Weak AI (Narrow AI)."
            ),
            question_type=QuestionType.FIGURE_CAPTION,
            gold_evidence=[
                GoldEvidence(
                    text=(
                        "caption “FIGURE 2. Weak AI vs Strong AI” plus the diagram labels. "
                        "The surrounding II.A text corroborates the Weak/Narrow and Strong/AGI relationship."
                    ),
                    element_id="sample_paper.pdf#page_2_cap_1",
                    page_number=2,
                    section="Figure 2 / II.A",
                    metadata={
                        "figure_caption": "FIGURE 2. Weak AI vs Strong AI",
                        "related_image_id": "sample_paper.pdf#page_2_img_2",
                    },
                ),
            ],
            special_notes=(
                "The sub-type breakdown exists only in the image. A text-only extraction can match just the "
                "caption and the II.A prose, so score this question separately for “found the caption” and "
                "“answered the sub-type detail.”"
            ),
        ),
        EvaluationQuestion(
            id="Q7",
            query="Which real-world example does the paper use to illustrate gender bias in AI, and in what year did it occur?",
            reference_answer=(
                "Amazon’s AI-based candidate screening, in 2018. "
                "A gender bias was reportedly found in its review of applicants."
            ),
            question_type=QuestionType.SPECIFIC_FACTUAL,
            gold_evidence=[
                GoldEvidence(
                    text=(
                        "...for example, the Amazon applicant scandal in 2018, where AI was used to screen "
                        "candidates, apparently, there was a gender bias found based on the review [21]."
                    ),
                    element_id="sample_paper.pdf#chunk_18",
                    page_number=3,
                    section="II.C.1, item 4",
                    metadata={"topic": "Amazon candidate screening gender bias 2018"},
                ),
            ],
            special_notes=None,
        ),
        EvaluationQuestion(
            id="Q8",
            query="Since when has Qrator Labs used a threat detection model against DDoS attacks, and how does it work?",
            reference_answer=(
                "Since 2006. The model learns patterns from the visitors using the website and "
                "uses them to filter DDoS attacks with very high accuracy."
            ),
            question_type=QuestionType.SPECIFIC_FACTUAL,
            gold_evidence=[
                GoldEvidence(
                    text=(
                        "Since 2006, Qrator labs have been using a threat detection model to protect themselves "
                        "against Distributed denial of service (DDOS) attacks based on the pattern learned from "
                        "the visitors using the website, which model allows filtering attacks with a very high accuracy [13]."
                    ),
                    element_id="sample_paper.pdf#chunk_14",
                    page_number=3,
                    section="II.B use cases",
                    metadata={"topic": "Qrator Labs threat detection model since 2006"},
                ),
            ],
            special_notes=None,
        ),
    ]


def get_retrieval_evaluation_dataset() -> EvaluationDataset:
    """Build and return the canonical retrieval evaluation dataset."""
    questions = get_retrieval_evaluation_questions()
    return EvaluationDataset(
        name="sample_paper_retrieval_evaluation_dataset",
        description=(
            "Curated evaluation dataset containing 8 benchmark questions for multimodal "
            "retrieval quality assessment based on data/evaluation/retrieval_questions.md."
        ),
        questions=questions,
    )
