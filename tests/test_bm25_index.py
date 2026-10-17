"""Comprehensive unit tests for the Day 15 sparse BM25 retrieval foundation.

Verifies:
- Text-like modality filtering (TEXT, CAPTION, TABLE, EQUATION)
- Strict IMAGE exclusion from the BM25 corpus
- Deterministic tokenization (case normalization, scientific identifiers, equations, tables)
- BM25 ranking (term frequency sensitivity, length normalization, multi-term queries)
- Element/provenance preservation (element_id, type, source_filename, page_number, metadata, relationships, content)
- Empty index and empty/whitespace query handling
- Parameter validations (k1, b, top_k, document type, element type)
- Adding a MultimodalDocument
- Fewer results than top_k
- No regression to existing FAISS/Multimodal components
"""

import pytest

from app.indexing.bm25_index import BM25Index, build_bm25_index, tokenize
from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.collection import MultimodalDocument, assemble_multimodal_document
from app.multimodal.models import (
    CaptionMatch,
    ExtractedCaption,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
)
from app.multimodal.representation import (
    ElementType,
    MultimodalElement,
    from_caption,
    from_equation,
    from_image,
    from_table,
)


# ===========================================================================
# 1. Deterministic Tokenization Tests
# ===========================================================================


class TestBM25Tokenization:
    """Verifies deterministic, lightweight tokenization for scientific multimodal text."""

    def test_case_normalization(self):
        tokens = tokenize("Attention Is All You Need: TRANSFORMER Architecture")
        assert tokens == [
            "attention",
            "is",
            "all",
            "you",
            "need",
            "transformer",
            "architecture",
        ]

    def test_punctuation_stripping(self):
        tokens = tokenize("Hello, World! (This is a test: [1, 2, 3]).")
        assert tokens == ["hello", "world", "this", "is", "a", "test", "1", "2", "3"]

    def test_scientific_subscripts_and_identifiers(self):
        tokens = tokenize("Variables d_k and W_q with dimension d_model = 512.")
        assert "d_k" in tokens
        assert "w_q" in tokens
        assert "d_model" in tokens
        assert "512" in tokens

    def test_hyphenated_terms_preservation(self):
        tokens = tokenize("The BLEU-4 score of scaled dot-product attention on WMT-2014.")
        assert "bleu-4" in tokens
        assert "dot-product" in tokens
        assert "wmt-2014" in tokens

    def test_equation_latex_terms(self):
        equation_str = r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V"
        tokens = tokenize(equation_str)
        assert "attention" in tokens
        assert "softmax" in tokens
        assert "frac" in tokens
        assert "sqrt" in tokens
        assert "d_k" in tokens
        assert "q" in tokens
        assert "k" in tokens
        assert "v" in tokens

    def test_table_markdown_terms(self):
        table_str = "| Model | BLEU-4 | Parameters |\n| Transformer-base | 27.3 | 65M |"
        tokens = tokenize(table_str)
        assert "model" in tokens
        assert "bleu-4" in tokens
        assert "parameters" in tokens
        assert "transformer-base" in tokens
        assert "65m" in tokens

    def test_empty_and_whitespace_input(self):
        assert tokenize("") == []
        assert tokenize("   \t\n  ") == []
        assert tokenize(None) == []

    def test_non_string_input(self):
        assert tokenize(123) == []
        assert tokenize(["word"]) == []


# ===========================================================================
# 2. Modality Filtering & Strict IMAGE Exclusion Tests
# ===========================================================================


class TestModalityFiltering:
    """Verifies that only TEXT-like elements are indexed and IMAGE elements are excluded."""

    @pytest.fixture
    def sample_elements(self):
        text_el = MultimodalElement(
            element_id="txt-1",
            element_type=ElementType.TEXT,
            source_filename="paper.pdf",
            page_number=1,
            content="We propose the Transformer model based on multi-head attention.",
        )
        caption_el = MultimodalElement(
            element_id="cap-1",
            element_type=ElementType.CAPTION,
            source_filename="paper.pdf",
            page_number=2,
            content="Figure 1: The Transformer model architecture schema.",
            related_element_ids=["img-1"],
        )
        table_el = MultimodalElement(
            element_id="tbl-1",
            element_type=ElementType.TABLE,
            source_filename="paper.pdf",
            page_number=3,
            content="| Model | BLEU-4 |\n| Baseline | 28.4 |",
        )
        equation_el = MultimodalElement(
            element_id="eq-1",
            element_type=ElementType.EQUATION,
            source_filename="paper.pdf",
            page_number=4,
            content=r"\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)",
        )
        image_el = MultimodalElement(
            element_id="img-1",
            element_type=ElementType.IMAGE,
            source_filename="paper.pdf",
            page_number=2,
            content=b"\x89PNG\r\n\x1a\nFakeImageData",
            related_element_ids=["cap-1"],
            metadata={"image_path": "/data/fig1.png"},
        )
        return {
            "text": text_el,
            "caption": caption_el,
            "table": table_el,
            "equation": equation_el,
            "image": image_el,
        }

    def test_text_like_modalities_indexed(self, sample_elements):
        index = BM25Index()
        text_like = [
            sample_elements["text"],
            sample_elements["caption"],
            sample_elements["table"],
            sample_elements["equation"],
        ]
        index.add_elements(text_like)
        assert index.total_elements == 4
        assert len(index) == 4

        # All four are retrievable
        assert index.get_element_by_id("txt-1") is not None
        assert index.get_element_by_id("cap-1") is not None
        assert index.get_element_by_id("tbl-1") is not None
        assert index.get_element_by_id("eq-1") is not None

    def test_image_elements_strictly_excluded(self, sample_elements):
        index = BM25Index()
        # Add only an image element
        index.add_elements([sample_elements["image"]])
        assert index.total_elements == 0
        assert len(index) == 0
        assert index.get_element_by_id("img-1") is None

    def test_mixed_collection_filters_images(self, sample_elements):
        index = BM25Index()
        # Add all 5 elements (4 text-like, 1 image)
        all_elements = list(sample_elements.values())
        index.add_elements(all_elements)

        assert index.total_elements == 4
        assert index.get_element_by_id("img-1") is None
        assert index.get_element_by_id("txt-1") is not None
        assert index.get_element_by_id("cap-1") is not None
        assert index.get_element_by_id("tbl-1") is not None
        assert index.get_element_by_id("eq-1") is not None


# ===========================================================================
# 3. Validation Tests (k1, b, top_k, types)
# ===========================================================================


class TestBM25Validation:
    """Verifies parameter validation and error handling."""

    def test_default_parameters(self):
        index = BM25Index()
        assert index.k1 == 1.5
        assert index.b == 0.75

    def test_custom_valid_parameters(self):
        index = BM25Index(k1=1.2, b=0.8)
        assert index.k1 == 1.2
        assert index.b == 0.8

    def test_negative_k1_raises_value_error(self):
        with pytest.raises(ValueError, match="k1 must be non-negative"):
            BM25Index(k1=-0.1)

    def test_invalid_b_range_raises_value_error(self):
        with pytest.raises(ValueError, match="b must be between 0.0 and 1.0"):
            BM25Index(b=-0.01)
        with pytest.raises(ValueError, match="b must be between 0.0 and 1.0"):
            BM25Index(b=1.01)

    def test_non_numeric_parameters_raise_type_error(self):
        with pytest.raises(TypeError, match="k1 must be a float or int"):
            BM25Index(k1="fast")
        with pytest.raises(TypeError, match="k1 must be a float or int"):
            BM25Index(k1=True)  # bool is subclass of int in python
        with pytest.raises(TypeError, match="b must be a float or int"):
            BM25Index(b="high")
        with pytest.raises(TypeError, match="b must be a float or int"):
            BM25Index(b=False)

    def test_invalid_top_k_raises_value_error(self):
        index = BM25Index()
        index.add_elements(
            [
                MultimodalElement(
                    element_id="el-1",
                    element_type=ElementType.TEXT,
                    source_filename="test.pdf",
                    page_number=1,
                    content="Sample text content",
                )
            ]
        )
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            index.search("sample", top_k=0)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            index.search("sample", top_k=-5)

    def test_invalid_element_type_in_add_elements(self):
        index = BM25Index()
        with pytest.raises(TypeError, match="must be a MultimodalElement instance"):
            index.add_elements(["not an element"])

    def test_none_elements_raises_value_error(self):
        index = BM25Index()
        with pytest.raises(ValueError, match="elements cannot be None"):
            index.add_elements(None)

    def test_invalid_document_raises_type_error(self):
        index = BM25Index()
        with pytest.raises(ValueError, match="document cannot be None"):
            index.add_document(None)
        with pytest.raises(TypeError, match="must have an 'elements' attribute"):
            index.add_document("string is not a document")

    def test_duplicate_element_id_rejection(self):
        index = BM25Index()
        el = MultimodalElement(
            element_id="dup-1",
            element_type=ElementType.TEXT,
            source_filename="paper.pdf",
            page_number=1,
            content="First text version",
        )
        index.add_elements([el])
        with pytest.raises(ValueError, match="Duplicate element_id detected"):
            index.add_elements([el])


# ===========================================================================
# 4. Empty Index, Empty Query & Edge Cases
# ===========================================================================


class TestEmptyAndEdgeCases:
    """Verifies behavior on empty indices, empty queries, and zero-match cases."""

    def test_empty_index_returns_empty_list(self):
        index = BM25Index()
        assert index.total_elements == 0
        assert index.avgdl == 0.0
        results = index.search("transformer attention", top_k=5)
        assert results == []

    def test_empty_and_whitespace_query_returns_empty_list(self):
        index = BM25Index()
        index.add_elements(
            [
                MultimodalElement(
                    element_id="el-1",
                    element_type=ElementType.TEXT,
                    source_filename="test.pdf",
                    page_number=1,
                    content="The quick brown fox",
                )
            ]
        )
        assert index.search("") == []
        assert index.search("   \t\n  ") == []
        assert index.search(None) == []

    def test_invalid_query_type_raises_type_error(self):
        index = BM25Index()
        index.add_elements(
            [
                MultimodalElement(
                    element_id="el-1",
                    element_type=ElementType.TEXT,
                    source_filename="test.pdf",
                    page_number=1,
                    content="The quick brown fox",
                )
            ]
        )
        with pytest.raises(TypeError, match="query must be a string"):
            index.search(12345)

    def test_query_with_no_matches_returns_empty_list(self):
        index = BM25Index()
        index.add_elements(
            [
                MultimodalElement(
                    element_id="el-1",
                    element_type=ElementType.TEXT,
                    source_filename="test.pdf",
                    page_number=1,
                    content="Convolutional neural networks for image processing",
                )
            ]
        )
        results = index.search("quantum teleportation superconductor", top_k=5)
        assert results == []

    def test_fewer_results_than_top_k(self):
        index = BM25Index()
        index.add_elements(
            [
                MultimodalElement(
                    element_id="el-1",
                    element_type=ElementType.TEXT,
                    source_filename="test.pdf",
                    page_number=1,
                    content="Natural language processing with transformers",
                ),
                MultimodalElement(
                    element_id="el-2",
                    element_type=ElementType.TEXT,
                    source_filename="test.pdf",
                    page_number=1,
                    content="Computer vision with convolutional networks",
                ),
            ]
        )
        # Only el-1 matches "transformers"
        results = index.search("transformers", top_k=10)
        assert len(results) == 1
        assert results[0].element_id == "el-1"

    def test_clear_method_resets_index(self):
        index = BM25Index()
        index.add_elements(
            [
                MultimodalElement(
                    element_id="el-1",
                    element_type=ElementType.TEXT,
                    source_filename="test.pdf",
                    page_number=1,
                    content="Recurrent neural network memory",
                )
            ]
        )
        assert index.total_elements == 1
        assert index.avgdl > 0

        index.clear()
        assert index.total_elements == 0
        assert index.avgdl == 0.0
        assert index.elements == []
        assert index.get_element_by_id("el-1") is None
        assert index.search("recurrent") == []


# ===========================================================================
# 5. BM25 Ranking and Scoring Tests
# ===========================================================================


class TestBM25Ranking:
    """Verifies that BM25 scoring adheres to term frequency saturation, length normalization, and ranking."""

    def test_term_frequency_sensitivity(self):
        """Document with higher term frequency (repeated term) should score higher."""
        index = BM25Index()
        el_low_tf = MultimodalElement(
            element_id="low-tf",
            element_type=ElementType.TEXT,
            source_filename="doc.pdf",
            page_number=1,
            content="The model uses attention mechanisms for translation.",
        )
        el_high_tf = MultimodalElement(
            element_id="high-tf",
            element_type=ElementType.TEXT,
            source_filename="doc.pdf",
            page_number=1,
            content="Attention mechanisms attention multi-head attention self-attention.",
        )
        index.add_elements([el_low_tf, el_high_tf])

        results = index.search("attention", top_k=2)
        assert len(results) == 2
        assert results[0].element_id == "high-tf"
        assert results[0].score > results[1].score

    def test_length_normalization(self):
        """With identical matching term count, the shorter document should rank higher due to b parameter."""
        index = BM25Index(k1=1.5, b=0.75)
        # Short document: 1 occurrence of 'hyperparameter' in 5 words
        short_doc = MultimodalElement(
            element_id="short-doc",
            element_type=ElementType.TEXT,
            source_filename="doc.pdf",
            page_number=1,
            content="Critical hyperparameter tuning process details.",
        )
        # Long document: 1 occurrence of 'hyperparameter' padded with 50 unrelated words
        padding = " " + " ".join(["word" + str(i) for i in range(50)])
        long_doc = MultimodalElement(
            element_id="long-doc",
            element_type=ElementType.TEXT,
            source_filename="doc.pdf",
            page_number=2,
            content=f"Hyperparameter analysis with extensive experimentation{padding}.",
        )
        index.add_elements([short_doc, long_doc])

        results = index.search("hyperparameter", top_k=2)
        assert len(results) == 2
        assert results[0].element_id == "short-doc"
        assert results[0].score > results[1].score

    def test_multi_term_query_ranking(self):
        """Document matching all query terms should score higher than document matching only one."""
        index = BM25Index()
        partial_match = MultimodalElement(
            element_id="partial",
            element_type=ElementType.TEXT,
            source_filename="doc.pdf",
            page_number=1,
            content="The algorithm achieves high accuracy on benchmark datasets.",
        )
        full_match = MultimodalElement(
            element_id="full",
            element_type=ElementType.TEXT,
            source_filename="doc.pdf",
            page_number=2,
            content="Scaled dot-product attention computes compatibility between queries and keys.",
        )
        index.add_elements([partial_match, full_match])

        results = index.search("scaled dot-product attention", top_k=2)
        assert len(results) >= 1
        assert results[0].element_id == "full"

    def test_equation_specific_sparse_retrieval(self):
        """Sparse retrieval on mathematical variable 'd_k' finds the equation element."""
        index = BM25Index()
        text_el = MultimodalElement(
            element_id="txt-intro",
            element_type=ElementType.TEXT,
            source_filename="transformer.pdf",
            page_number=1,
            content="Introduction to sequence transduction models.",
        )
        eq_el = MultimodalElement(
            element_id="eq-attention",
            element_type=ElementType.EQUATION,
            source_filename="transformer.pdf",
            page_number=3,
            content=r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V",
        )
        index.add_elements([text_el, eq_el])

        results = index.search("d_k", top_k=2)
        assert len(results) == 1
        assert results[0].element_id == "eq-attention"
        assert results[0].element_type == ElementType.EQUATION

    def test_table_specific_sparse_retrieval(self):
        """Sparse retrieval on 'bleu-4' finds the table element."""
        index = BM25Index()
        text_el = MultimodalElement(
            element_id="txt-results",
            element_type=ElementType.TEXT,
            source_filename="transformer.pdf",
            page_number=5,
            content="Results on English-to-German and English-to-French translation tasks.",
        )
        table_el = MultimodalElement(
            element_id="tbl-results",
            element_type=ElementType.TABLE,
            source_filename="transformer.pdf",
            page_number=5,
            content="| Model | BLEU-4 | PPL |\n| Transformer (big) | 28.4 | 4.1 |",
        )
        index.add_elements([text_el, table_el])

        results = index.search("bleu-4", top_k=2)
        assert len(results) == 1
        assert results[0].element_id == "tbl-results"
        assert results[0].element_type == ElementType.TABLE


# ===========================================================================
# 6. Element & Provenance Preservation Tests
# ===========================================================================


class TestProvenancePreservation:
    """Verifies that retrieval results preserve original MultimodalElement fields intact."""

    def test_result_structure_and_provenance(self):
        index = BM25Index()
        original_el = MultimodalElement(
            element_id="provenance-el-42",
            element_type=ElementType.TABLE,
            source_filename="scientific_paper_2024.pdf",
            page_number=7,
            content="| Accuracy | F1 |\n| 94.2% | 0.93 |",
            related_element_ids=["cap-table-1"],
            metadata={"num_rows": 2, "headers": ["Accuracy", "F1"]},
        )
        index.add_elements([original_el])

        results = index.search("accuracy f1", top_k=1)
        assert len(results) == 1
        res = results[0]

        # Verify it returns the standard MultimodalSearchResult
        assert isinstance(res, MultimodalSearchResult)

        # Exact object preservation
        assert res.element is original_el

        # Direct accessors preservation
        assert res.element_id == "provenance-el-42"
        assert res.element_type == ElementType.TABLE
        assert res.source_filename == "scientific_paper_2024.pdf"
        assert res.page_number == 7
        assert res.related_element_ids == ["cap-table-1"]
        assert res.content == "| Accuracy | F1 |\n| 94.2% | 0.93 |"
        assert res.metadata == {"num_rows": 2, "headers": ["Accuracy", "F1"]}

        # Score is valid positive float
        assert isinstance(res.score, float)
        assert res.score > 0.0


# ===========================================================================
# 7. MultimodalDocument and Factory Function Tests
# ===========================================================================


class TestDocumentAndFactory:
    """Verifies integration with MultimodalDocument and factory constructors."""

    @pytest.fixture
    def sample_multimodal_document(self):
        images = [
            ExtractedImage(
                image_id="img-schema",
                source_filename="paper.pdf",
                page_number=2,
                image_index=0,
                image_format="png",
                image_data=b"\x89PNG\r\n\x1a\nFakeImageBytes",
                width=100,
                height=100,
            )
        ]
        captions = [
            ExtractedCaption(
                caption_id="cap-schema",
                source_filename="paper.pdf",
                page_number=2,
                caption_index=0,
                caption_type="figure",
                text="Figure 1: Overall model architecture diagram.",
                label="Figure 1",
            )
        ]
        tables = [
            ExtractedTable(
                table_id="tbl-benchmark",
                source_filename="paper.pdf",
                page_number=4,
                table_index=0,
                num_rows=2,
                num_cols=2,
                rows=[["Model", "Score"], ["Ours", "99.1"]],
                headers=["Model", "Score"],
                content="| Model | Score |\n| Ours | 99.1 |",
            )
        ]
        equations = [
            ExtractedEquation(
                equation_id="eq-loss",
                source_filename="paper.pdf",
                page_number=3,
                equation_index=0,
                text=r"\mathcal{L} = -\sum y \log(\hat{y})",
                is_display=True,
            )
        ]
        matches = [
            CaptionMatch(
                element_id="img-schema",
                caption_id="cap-schema",
                caption_type="figure",
                proximity_pts=10.0,
                source_filename="paper.pdf",
                page_number=2,
            )
        ]
        return assemble_multimodal_document(
            images=images,
            captions=captions,
            tables=tables,
            equations=equations,
            caption_matches=matches,
            source_filename="paper.pdf",
        )

    def test_add_document_filters_images(self, sample_multimodal_document):
        index = BM25Index()
        index.add_document(sample_multimodal_document)

        # The document has 1 image, 1 caption, 1 table, 1 equation = 4 total elements
        # Only 3 text-like elements should enter the BM25 index
        assert index.total_elements == 3
        assert index.get_element_by_id("img-schema") is None
        assert index.get_element_by_id("cap-schema") is not None
        assert index.get_element_by_id("tbl-benchmark") is not None
        assert index.get_element_by_id("eq-loss") is not None

        # Search retrieves caption
        results = index.search("architecture diagram")
        assert len(results) >= 1
        assert results[0].element_id == "cap-schema"

    def test_from_document_factory(self, sample_multimodal_document):
        index = BM25Index.from_document(sample_multimodal_document, k1=1.6, b=0.7)
        assert index.total_elements == 3
        assert index.k1 == 1.6
        assert index.b == 0.7

    def test_from_elements_factory(self):
        el = MultimodalElement(
            element_id="el-fact",
            element_type=ElementType.TEXT,
            source_filename="paper.pdf",
            page_number=1,
            content="Factory constructor testing text.",
        )
        index = BM25Index.from_elements([el], k1=1.3, b=0.8)
        assert index.total_elements == 1
        assert index.k1 == 1.3
        assert index.b == 0.8

    def test_build_bm25_index_helper(self, sample_multimodal_document):
        index_from_doc = build_bm25_index(sample_multimodal_document)
        assert isinstance(index_from_doc, BM25Index)
        assert index_from_doc.total_elements == 3

        elements = [
            MultimodalElement(
                element_id="el-1",
                element_type=ElementType.TEXT,
                source_filename="test.pdf",
                page_number=1,
                content="Text content",
            )
        ]
        index_from_seq = build_bm25_index(elements)
        assert isinstance(index_from_seq, BM25Index)
        assert index_from_seq.total_elements == 1

        with pytest.raises(TypeError, match="must be a MultimodalDocument or Sequence"):
            build_bm25_index(12345)


# ===========================================================================
# 8. Non-Regression Verification
# ===========================================================================


def test_existing_faiss_and_multimodal_models_untouched():
    """Confirms that importing and using FAISSIndex, MultimodalFAISSIndex, and MultimodalSearchResult is unaffected."""
    from app.indexing import (
        FAISSIndex,
        ModalityFAISSIndex,
        MultimodalFAISSIndex,
        MultimodalSearchResult,
        build_index,
        build_multimodal_index,
    )

    assert FAISSIndex is not None
    assert ModalityFAISSIndex is not None
    assert MultimodalFAISSIndex is not None
    assert MultimodalSearchResult is not None
    assert build_index is not None
    assert build_multimodal_index is not None
