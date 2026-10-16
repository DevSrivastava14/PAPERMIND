"""Focused tests for Day 14: Multimodal retrieved-context modeling and assembly."""

from typing import List
import pytest

from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.collection import MultimodalDocument, assemble_multimodal_document
from app.multimodal.models import (
    CaptionMatch,
    ExtractedCaption,
    ExtractedImage,
    ExtractedTable,
)
from app.multimodal.representation import ElementType, MultimodalElement
from app.retrieval.multimodal_assembler import (
    MultimodalContextAssembler,
    assemble_multimodal_context,
)
from app.retrieval.multimodal_models import (
    MultimodalCitation,
    MultimodalRetrievedContext,
)


# ===========================================================================
# Fixtures and Helpers
# ===========================================================================


@pytest.fixture
def sample_text_element() -> MultimodalElement:
    return MultimodalElement(
        element_id="txt_intro",
        element_type=ElementType.TEXT,
        source_filename="transformer.pdf",
        page_number=1,
        content="The Transformer is the first transduction model relying entirely on self-attention.",
        metadata={"section": "Abstract"},
    )


@pytest.fixture
def sample_caption_element() -> MultimodalElement:
    return MultimodalElement(
        element_id="cap_arch",
        element_type=ElementType.CAPTION,
        source_filename="transformer.pdf",
        page_number=3,
        content="Figure 1: The Transformer - model architecture.",
        related_element_ids=["img_arch"],
        metadata={"label": "Figure 1", "caption_type": "figure"},
    )


@pytest.fixture
def sample_table_element() -> MultimodalElement:
    return MultimodalElement(
        element_id="tbl_bleu",
        element_type=ElementType.TABLE,
        source_filename="transformer.pdf",
        page_number=8,
        content="| Model | BLEU (EN-DE) |\n|---|---|\n| Transformer (base) | 27.3 |\n| Transformer (big) | 28.4 |",
        metadata={"rows": 3, "cols": 2},
    )


@pytest.fixture
def sample_equation_element() -> MultimodalElement:
    return MultimodalElement(
        element_id="eq_attn",
        element_type=ElementType.EQUATION,
        source_filename="transformer.pdf",
        page_number=4,
        content="Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V",
        metadata={"equation_type": "display"},
    )


@pytest.fixture
def sample_image_element() -> MultimodalElement:
    return MultimodalElement(
        element_id="img_arch",
        element_type=ElementType.IMAGE,
        source_filename="transformer.pdf",
        page_number=3,
        content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRfakeimagebytes",
        related_element_ids=["cap_arch"],
        metadata={
            "image_format": "png",
            "width": 800,
            "height": 600,
        },
    )


# ===========================================================================
# Test Cases
# ===========================================================================


class TestMultimodalAssembly:
    """Tests for MultimodalContextAssembler and MultimodalRetrievedContext."""

    def test_text_formatting(self, sample_text_element: MultimodalElement) -> None:
        """Requirement 1: Text formatting produces readable evidence."""
        res = MultimodalSearchResult(
            element=sample_text_element,
            score=0.92,
        )
        assembler = MultimodalContextAssembler()
        context = assembler.assemble([res])

        assert len(context) == 1
        assert "[1]" in context.formatted_context
        assert "[Text]" in context.formatted_context
        assert "transformer.pdf" in context.formatted_context
        assert "Page 1" in context.formatted_context
        assert "The Transformer is the first transduction model" in context.formatted_context

    def test_caption_formatting(self, sample_caption_element: MultimodalElement) -> None:
        """Requirement 2: Caption formatting produces caption evidence."""
        res = MultimodalSearchResult(
            element=sample_caption_element,
            score=0.85,
        )
        context = assemble_multimodal_context([res])

        assert len(context) == 1
        assert "[1]" in context.formatted_context
        assert "[Caption]" in context.formatted_context
        assert "transformer.pdf" in context.formatted_context
        assert "Page 3" in context.formatted_context
        assert "Figure 1: The Transformer - model architecture." in context.formatted_context

    def test_table_formatting(self, sample_table_element: MultimodalElement) -> None:
        """Requirement 3: Table formatting produces table markdown/content."""
        res = MultimodalSearchResult(
            element=sample_table_element,
            score=0.78,
        )
        assembler = MultimodalContextAssembler()
        context = assembler.assemble([res])

        assert len(context) == 1
        assert "[1]" in context.formatted_context
        assert "[Table]" in context.formatted_context
        assert "Page 8" in context.formatted_context
        assert "| Model | BLEU (EN-DE) |" in context.formatted_context
        assert "Transformer (big) | 28.4" in context.formatted_context

    def test_equation_formatting(self, sample_equation_element: MultimodalElement) -> None:
        """Requirement 4: Equation formatting produces formula content."""
        res = MultimodalSearchResult(
            element=sample_equation_element,
            score=0.88,
        )
        context = assemble_multimodal_context([res])

        assert len(context) == 1
        assert "[1]" in context.formatted_context
        assert "[Equation]" in context.formatted_context
        assert "Page 4" in context.formatted_context
        assert "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V" in context.formatted_context

    def test_image_formatting_no_raw_bytes(
        self, sample_image_element: MultimodalElement
    ) -> None:
        """Requirement 5: Image formatting produces descriptor and NEVER includes raw bytes."""
        res = MultimodalSearchResult(
            element=sample_image_element,
            score=0.95,
        )
        assembler = MultimodalContextAssembler()
        context = assembler.assemble([res])

        assert len(context) == 1
        assert "[1]" in context.formatted_context
        assert "[Image]" in context.formatted_context
        assert "[Image: img_arch]" in context.formatted_context
        assert "format: png" in context.formatted_context
        assert "dimensions: 800x600" in context.formatted_context

        # CRITICAL: Verify raw bytes are not present in formatted context string
        assert "fakeimagebytes" not in context.formatted_context
        assert "\\x89PNG" not in context.formatted_context
        assert "\x89PNG" not in context.formatted_context

    def test_sequential_citation_labels(
        self,
        sample_text_element: MultimodalElement,
        sample_caption_element: MultimodalElement,
        sample_table_element: MultimodalElement,
        sample_image_element: MultimodalElement,
    ) -> None:
        """Requirement 6: Citations and headers increment sequentially [1], [2], [3], [4]."""
        results = [
            MultimodalSearchResult(sample_text_element, score=0.9),
            MultimodalSearchResult(sample_caption_element, score=0.8),
            MultimodalSearchResult(sample_table_element, score=0.7),
            MultimodalSearchResult(sample_image_element, score=0.6),
        ]
        context = assemble_multimodal_context(results)

        assert len(context.citations) == 4
        assert [c.citation_label for c in context.citations] == ["[1]", "[2]", "[3]", "[4]"]

        # Check in formatted text
        lines = context.formatted_context.split("\n\n")
        assert len(lines) == 4
        assert lines[0].startswith("[1] [Text]")
        assert lines[1].startswith("[2] [Caption]")
        assert lines[2].startswith("[3] [Table]")
        assert lines[3].startswith("[4] [Image]")

    def test_source_filename_and_page_preservation(
        self,
        sample_text_element: MultimodalElement,
        sample_table_element: MultimodalElement,
    ) -> None:
        """Requirement 7: Source filename and 1-based page numbers are exactly preserved."""
        results = [
            MultimodalSearchResult(sample_text_element, score=0.91),
            MultimodalSearchResult(sample_table_element, score=0.81),
        ]
        context = assemble_multimodal_context(results)

        cit1 = context.citations[0]
        assert cit1.element_id == "txt_intro"
        assert cit1.element_type == ElementType.TEXT
        assert cit1.source_filename == "transformer.pdf"
        assert cit1.page_number == 1
        assert cit1.score == 0.91

        cit2 = context.citations[1]
        assert cit2.element_id == "tbl_bleu"
        assert cit2.element_type == ElementType.TABLE
        assert cit2.source_filename == "transformer.pdf"
        assert cit2.page_number == 8
        assert cit2.score == 0.81

    def test_relationship_and_caption_exposure_for_image(
        self,
        sample_image_element: MultimodalElement,
        sample_caption_element: MultimodalElement,
    ) -> None:
        """Requirement 8: Related element IDs exposed and linked caption included via MultimodalDocument."""
        doc = MultimodalDocument(
            elements=[sample_image_element, sample_caption_element],
            source_filename="transformer.pdf",
        )

        res = MultimodalSearchResult(
            element=sample_image_element,
            score=0.94,
        )

        # 1. Assembling with document resolves linked caption
        assembler = MultimodalContextAssembler(document=doc)
        context_with_doc = assembler.assemble([res])

        # Exposed related_element_ids
        assert context_with_doc.related_element_ids == ["cap_arch"]

        # Text formatted includes resolved caption
        assert "Caption: Figure 1: The Transformer - model architecture." in context_with_doc.formatted_context

        # 2. Assembling without document exposes IDs without inventing caption text
        assembler_no_doc = MultimodalContextAssembler()
        context_no_doc = assembler_no_doc.assemble([res])
        assert context_no_doc.related_element_ids == ["cap_arch"]
        assert "Related Element IDs: cap_arch" in context_no_doc.formatted_context
        assert "Caption:" not in context_no_doc.formatted_context

    def test_empty_results_handling(self) -> None:
        """Requirement 9: Empty search results produce clean empty context."""
        assembler = MultimodalContextAssembler()
        context = assembler.assemble([])

        assert len(context) == 0
        assert context.formatted_context == ""
        assert context.citations == []
        assert context.search_results == []
        assert context.elements == []
        assert context.visual_results == []
        assert context.text_results == []
        assert context.related_element_ids == []

    def test_preservation_of_original_search_results(
        self,
        sample_text_element: MultimodalElement,
        sample_image_element: MultimodalElement,
    ) -> None:
        """Requirement 10: Original MultimodalSearchResult objects are preserved."""
        res1 = MultimodalSearchResult(sample_text_element, score=0.9)
        res2 = MultimodalSearchResult(sample_image_element, score=0.85)

        context = assemble_multimodal_context([res1, res2])

        assert len(context.search_results) == 2
        assert context.search_results[0] is res1
        assert context.search_results[1] is res2
        assert context.elements[0] is sample_text_element
        assert context.elements[1] is sample_image_element

    def test_context_helper_properties(
        self,
        sample_text_element: MultimodalElement,
        sample_table_element: MultimodalElement,
        sample_image_element: MultimodalElement,
    ) -> None:
        """Verifies visual_results, text_results, and citation lookup helpers."""
        res_text = MultimodalSearchResult(sample_text_element, score=0.9)
        res_table = MultimodalSearchResult(sample_table_element, score=0.8)
        res_img = MultimodalSearchResult(sample_image_element, score=0.7)

        context = assemble_multimodal_context([res_text, res_table, res_img])

        assert len(context.text_results) == 2
        assert res_text in context.text_results
        assert res_table in context.text_results

        assert len(context.visual_results) == 1
        assert context.visual_results[0] is res_img

        # Citation lookup
        cit = context.get_citation_by_label("[2]")
        assert cit is not None
        assert cit.element_id == "tbl_bleu"
        assert context.get_citation_by_label("[99]") is None

        # Result lookup
        hit = context.get_result_by_element_id("img_arch")
        assert hit is not None
        assert hit.element_type == ElementType.IMAGE
        assert context.get_result_by_element_id("non_existent") is None

    def test_citation_validation(self) -> None:
        """Verifies validation rules on MultimodalCitation."""
        with pytest.raises(ValueError, match="citation_label cannot be empty"):
            MultimodalCitation(
                citation_label="",
                element_id="el_1",
                element_type=ElementType.TEXT,
                source_filename="doc.pdf",
                page_number=1,
            )

        with pytest.raises(ValueError, match="element_id cannot be empty"):
            MultimodalCitation(
                citation_label="[1]",
                element_id="",
                element_type=ElementType.TEXT,
                source_filename="doc.pdf",
                page_number=1,
            )

        with pytest.raises(ValueError, match="source_filename cannot be empty"):
            MultimodalCitation(
                citation_label="[1]",
                element_id="el_1",
                element_type=ElementType.TEXT,
                source_filename="",
                page_number=1,
            )

        with pytest.raises(ValueError, match="page_number must be an integer >= 1"):
            MultimodalCitation(
                citation_label="[1]",
                element_id="el_1",
                element_type=ElementType.TEXT,
                source_filename="doc.pdf",
                page_number=0,
            )
