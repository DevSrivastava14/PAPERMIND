"""End-to-end multimodal retrieval integration tests for Day 14.

Verifies the complete flow from heterogeneous multimodal paper elements:
MultimodalDocument
      ↓
MultimodalEmbedder (384-d text, 512-d visual)
      ↓
MultimodalFAISSIndex (text_index: 384-d, visual_index: 512-d)
      ↓
MultimodalRetrievalPipeline
      ↓
retrieve_and_assemble()
      ↓
MultimodalRetrievedContext (formatted grounded context + citations + original results)
"""

from typing import Any, List, Optional
import numpy as np
import pytest

from app.chunking.models import TextChunk
from app.embeddings.base import BaseEmbedder
from app.indexing.multimodal_faiss_index import (
    MultimodalFAISSIndex,
    build_multimodal_index,
)
from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.collection import MultimodalDocument, assemble_multimodal_document
from app.multimodal.embedding import (
    BaseVisualEmbedder,
    MultimodalEmbedder,
    get_multimodal_embedder,
    reset_multimodal_embedder,
)
from app.multimodal.models import (
    CaptionMatch,
    ExtractedCaption,
    ExtractedEquation,
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
from app.retrieval.multimodal_pipeline import (
    MultimodalRetrievalPipeline,
    build_multimodal_pipeline,
)


# ===========================================================================
# Deterministic Mock Embedders
# ===========================================================================


class DeterministicTextEmbedder(BaseEmbedder):
    """Deterministic 384-d mock text embedder routing semantic concepts to orthogonal dimensions."""

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "mock-text-mini-384"

    def embed_texts(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            lower = text.lower()
            if "bleu" in lower or "en-de" in lower:
                vec[1] = 1.0
            elif "softmax" in lower or "qk" in lower or "sqrt" in lower:
                vec[2] = 1.0
            elif "figure 1" in lower or "schematic" in lower:
                vec[3] = 1.0
            elif "recurrent" in lower or "self-attention" in lower or "mechanism" in lower:
                vec[0] = 1.0
            else:
                # Deterministic fallback based on hash
                dim = (hash(text) % 380) + 4
                vec[dim] = 1.0
            vectors.append(vec)

        return np.vstack(vectors).astype(np.float32)


class DeterministicVisualEmbedder(BaseVisualEmbedder):
    """Deterministic 512-d mock visual embedder routing image signatures to orthogonal dimensions."""

    @property
    def dimension(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return "mock-clip-vit-512"

    def embed_images(
        self, images: List[Any], batch_size: Optional[int] = None
    ) -> np.ndarray:
        if not images:
            return np.empty((0, 512), dtype=np.float32)

        vectors = []
        for img in images:
            vec = np.zeros(512, dtype=np.float32)
            if isinstance(img, (bytes, bytearray)):
                if b"architecture" in img:
                    vec[0] = 1.0
                elif b"results" in img:
                    vec[1] = 1.0
                else:
                    vec[img[0] % 512 if len(img) > 0 else 0] = 1.0
            elif isinstance(img, str):
                vec[hash(img) % 512] = 1.0
            else:
                vec[0] = 1.0
            vectors.append(vec)

        return np.vstack(vectors).astype(np.float32)


# ===========================================================================
# Fixture: Complete Multimodal Document
# ===========================================================================


@pytest.fixture
def multimodal_paper_document() -> MultimodalDocument:
    """Assembles a realistic scientific paper containing all 5 multimodal element types."""
    # 1. Text chunk
    chunk = TextChunk(
        chunk_id="chunk_sec1",
        text="The Transformer replaces recurrent layers with multi-head self-attention mechanisms.",
        source_filename="attention_paper.pdf",
        chunk_index=0,
        page_numbers=[1],
    )

    # 2. Extracted Image
    img = ExtractedImage(
        image_id="img_arch_fig1",
        source_filename="attention_paper.pdf",
        page_number=2,
        image_index=0,
        image_format="png",
        image_data=b"\x89PNG\r\n\x1a\narchitecture_diagram_data",
        width=400,
        height=300,
    )


    # 3. Extracted Caption
    cap = ExtractedCaption(
        caption_id="cap_arch_fig1",
        source_filename="attention_paper.pdf",
        page_number=2,
        caption_index=0,
        caption_type="figure",
        label="Figure 1",
        text="Figure 1: The Transformer model architecture schematic.",
    )

    # 4. Bidirectional Caption Match linking Image and Caption
    match = CaptionMatch(
        element_id="img_arch_fig1",
        caption_id="cap_arch_fig1",
        caption_type="figure",
        proximity_pts=12.5,
        source_filename="attention_paper.pdf",
        page_number=2,
    )

    # 5. Extracted Table
    tbl = ExtractedTable(
        table_id="tbl_results_tab1",
        source_filename="attention_paper.pdf",
        page_number=5,
        table_index=0,
        markdown="| Model | BLEU (EN-DE) | BLEU (EN-FR) |\n|---|---|---|\n| Transformer (base) | 27.3 | 38.1 |\n| Transformer (big) | 28.4 | 41.8 |",
        num_rows=3,
        num_cols=3,
    )

    # 6. Extracted Equation
    eq = ExtractedEquation(
        equation_id="eq_scaled_dot_prod",
        source_filename="attention_paper.pdf",
        page_number=3,
        equation_index=0,
        text=r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
    )


    # Assemble into MultimodalDocument
    doc = assemble_multimodal_document(
        text_chunks=[chunk],
        images=[img],
        captions=[cap],
        tables=[tbl],
        equations=[eq],
        caption_matches=[match],
        source_filename="attention_paper.pdf",
    )
    return doc


# ===========================================================================
# End-to-End Multimodal Integration Tests
# ===========================================================================


class TestEndToEndMultimodalRetrieval:
    """Verifies complete end-to-end flow from document assembly to prompt-ready retrieved context."""

    def test_document_embedding_and_modality_partitioning(
        self,
        multimodal_paper_document: MultimodalDocument,
    ) -> None:
        """Requirement 1, 2, 3: Document embeds and partitions into 384-d text and 512-d visual indices."""
        embedder = MultimodalEmbedder(
            text_embedder=DeterministicTextEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )

        index = build_multimodal_index(
            source=multimodal_paper_document,
            embedder=embedder,
        )

        assert isinstance(index, MultimodalFAISSIndex)
        # 4 text-like elements (text, caption, table, equation) in text_index
        assert index.total_text_elements == 4
        assert index.text_index.dimension == 384

        # 1 visual element (image) in visual_index
        assert index.total_visual_elements == 1
        assert index.visual_index.dimension == 512

        # Total count
        assert index.total_elements == 5
        assert len(index) == 5

    def test_text_query_retrieval_and_assembly(
        self,
        multimodal_paper_document: MultimodalDocument,
    ) -> None:
        """Requirement 4, 6, 7, 9: Text query -> text FAISS -> MultimodalRetrievedContext with citations."""
        embedder = MultimodalEmbedder(
            text_embedder=DeterministicTextEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )
        index = build_multimodal_index(multimodal_paper_document, embedder=embedder)
        pipeline = build_multimodal_pipeline(
            index=index,
            text_embedder=DeterministicTextEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )

        # Query targeting attention text chunk
        context = pipeline.retrieve_and_assemble(
            query="self-attention transformer mechanism",
            modality="text",
            top_k=2,
            document=multimodal_paper_document,
        )

        assert isinstance(context, MultimodalRetrievedContext)
        assert len(context) == 2
        assert len(context.citations) == 2
        assert len(context.search_results) == 2

        # Check top hit is the expected text chunk
        top_res = context.search_results[0]
        assert top_res.element_type == ElementType.TEXT
        assert top_res.element_id == "chunk_sec1"
        assert "The Transformer replaces recurrent layers" in top_res.element.text

        # Verify citation preservation
        top_cit = context.citations[0]
        assert top_cit.citation_label == "[1]"
        assert top_cit.element_id == "chunk_sec1"
        assert top_cit.element_type == ElementType.TEXT
        assert top_cit.source_filename == "attention_paper.pdf"
        assert top_cit.page_number == 1
        assert top_cit.score is not None
        assert top_cit.score > 0.0

        second_cit = context.citations[1]
        assert second_cit.citation_label == "[2]"

        # Verify formatted context contains human-readable evidence
        assert "[1] [Text] Source: attention_paper.pdf (Page 1)" in context.formatted_context
        assert "multi-head self-attention mechanisms" in context.formatted_context

    def test_table_and_equation_retrieval_and_assembly(
        self,
        multimodal_paper_document: MultimodalDocument,
    ) -> None:
        """Requirement 4: Table and equation elements are retrieved and properly formatted."""
        embedder = MultimodalEmbedder(
            text_embedder=DeterministicTextEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )
        index = build_multimodal_index(multimodal_paper_document, embedder=embedder)
        pipeline = build_multimodal_pipeline(
            index=index,
            text_embedder=DeterministicTextEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )

        # Query targeting table
        tbl_context = pipeline.retrieve_and_assemble(
            query="BLEU score model evaluation table",
            modality="text",
            top_k=1,
            document=multimodal_paper_document,
        )
        assert len(tbl_context) == 1
        assert tbl_context.citations[0].element_type == ElementType.TABLE
        assert tbl_context.citations[0].page_number == 5
        assert "| Model | BLEU (EN-DE) |" in tbl_context.formatted_context

        # Query targeting equation
        eq_context = pipeline.retrieve_and_assemble(
            query="softmax QK matrix equation formula",
            modality="text",
            top_k=1,
            document=multimodal_paper_document,
        )
        assert len(eq_context) == 1
        assert eq_context.citations[0].element_type == ElementType.EQUATION
        assert eq_context.citations[0].page_number == 3
        assert r"\text{Attention}(Q, K, V)" in eq_context.formatted_context

    def test_visual_query_retrieval_and_assembly_with_caption_resolution(
        self,
        multimodal_paper_document: MultimodalDocument,
    ) -> None:
        """Requirement 5, 8, 9: Visual query retrieves image, resolves caption, and excludes raw bytes."""
        embedder = MultimodalEmbedder(
            text_embedder=DeterministicTextEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )
        index = build_multimodal_index(multimodal_paper_document, embedder=embedder)
        pipeline = build_multimodal_pipeline(
            index=index,
            text_embedder=DeterministicTextEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )

        # Query with image bytes containing the architecture signature
        query_image_bytes = b"\x89PNG\r\narchitecture_query"

        context = pipeline.retrieve_and_assemble(
            query=query_image_bytes,
            modality="visual",
            top_k=1,
            document=multimodal_paper_document,
        )

        assert isinstance(context, MultimodalRetrievedContext)
        assert len(context) == 1

        # Check visual results
        assert len(context.visual_results) == 1
        img_hit = context.visual_results[0]
        assert img_hit.element_id == "img_arch_fig1"
        assert img_hit.element_type == ElementType.IMAGE
        assert img_hit.page_number == 2

        # Check citations
        cit = context.citations[0]
        assert cit.citation_label == "[1]"
        assert cit.element_id == "img_arch_fig1"
        assert cit.element_type == ElementType.IMAGE
        assert cit.source_filename == "attention_paper.pdf"
        assert cit.page_number == 2
        assert pytest.approx(cit.score, abs=1e-4) == 1.0

        # Check relationship exposure
        assert context.related_element_ids == ["cap_arch_fig1"]

        # Check formatted context has resolved caption text from document
        assert "[1] [Image] Source: attention_paper.pdf (Page 2)" in context.formatted_context
        assert "[Image: img_arch_fig1]" in context.formatted_context
        assert "Caption: Figure 1: The Transformer model architecture schematic." in context.formatted_context

        # CRITICAL: Confirm raw image bytes are never emitted into formatted context
        assert "architecture_diagram_data" not in context.formatted_context
        assert "\\x89PNG" not in context.formatted_context
        assert "\x89PNG" not in context.formatted_context

    def test_end_to_end_modality_isolation(
        self,
        multimodal_paper_document: MultimodalDocument,
    ) -> None:
        """Requirement 10: Modality isolation is strictly enforced across the complete pipeline."""
        embedder = MultimodalEmbedder(
            text_embedder=DeterministicTextEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )
        index = build_multimodal_index(multimodal_paper_document, embedder=embedder)
        pipeline = build_multimodal_pipeline(
            index=index,
            text_embedder=DeterministicTextEmbedder(),
            visual_embedder=DeterministicVisualEmbedder(),
        )

        # Retrieve all available text results (top_k=10)
        text_context = pipeline.retrieve_and_assemble(
            query="all elements in document",
            modality="text",
            top_k=10,
        )
        # Should return all 4 text-like elements, and 0 visual elements
        assert len(text_context) == 4
        for res in text_context.search_results:
            assert res.element_type != ElementType.IMAGE
            assert res.element_type in {
                ElementType.TEXT,
                ElementType.CAPTION,
                ElementType.TABLE,
                ElementType.EQUATION,
            }

        # Retrieve all available visual results (top_k=10)
        visual_context = pipeline.retrieve_and_assemble(
            query=b"any_image_bytes",
            modality="visual",
            top_k=10,
        )
        # Should return only the 1 visual element, and 0 text-like elements
        assert len(visual_context) == 1
        for res in visual_context.search_results:
            assert res.element_type == ElementType.IMAGE
