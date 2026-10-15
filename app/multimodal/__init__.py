from app.multimodal.caption_extractor import (
    CaptionExtractionResult,
    CaptionExtractor,
    extract_captions,
)
from app.multimodal.equation_extractor import EquationExtractor, extract_equations
from app.multimodal.image_extractor import ImageExtractor, extract_images
from app.multimodal.models import (
    CAPTION_TYPE_FIGURE,
    CAPTION_TYPE_TABLE,
    CaptionElement,
    CaptionMatch,
    EquationElement,
    ExtractedCaption,
    ExtractedCaptionElement,
    ExtractedEquation,
    ExtractedFormula,
    ExtractedImage,
    ExtractedTable,
    ExtractedTableElement,
    ExtractedVisualElement,
    TableElement,
    VisualElement,
)
from app.multimodal.table_extractor import TableExtractor, extract_tables
from app.multimodal.representation import (
    ElementType,
    ELEMENT_TYPE_CAPTION,
    ELEMENT_TYPE_EQUATION,
    ELEMENT_TYPE_IMAGE,
    ELEMENT_TYPE_TABLE,
    ELEMENT_TYPE_TEXT,
    MultimodalElement,
    SUPPORTED_ELEMENT_TYPES,
    attach_caption_matches,
    from_caption,
    from_equation,
    from_image,
    from_table,
    from_text_chunk,
    to_multimodal_element,
)
from app.multimodal.embedding import (
    BaseMultimodalEmbedder,
    BaseVisualEmbedder,
    MultimodalEmbeddedElement,
    MultimodalEmbedder,
    MultimodalEmbeddingResult,
    SentenceTransformerVisualEmbedder,
    embed_multimodal_elements,
    get_multimodal_embedder,
    reset_multimodal_embedder,
)
from app.multimodal.collection import (
    MultimodalCollection,
    MultimodalDocument,
    assemble_multimodal_collection,
    assemble_multimodal_document,
)

__all__ = [
    # Image / visual
    "ExtractedImage",
    "ExtractedVisualElement",
    "VisualElement",
    "ImageExtractor",
    "extract_images",
    # Equations
    "ExtractedEquation",
    "EquationElement",
    "ExtractedFormula",
    "EquationExtractor",
    "extract_equations",
    # Tables
    "ExtractedTable",
    "TableElement",
    "ExtractedTableElement",
    "TableExtractor",
    "extract_tables",
    # Captions
    "CAPTION_TYPE_FIGURE",
    "CAPTION_TYPE_TABLE",
    "ExtractedCaption",
    "CaptionElement",
    "ExtractedCaptionElement",
    "CaptionMatch",
    "CaptionExtractionResult",
    "CaptionExtractor",
    "extract_captions",
    # Unified Representation
    "ElementType",
    "ELEMENT_TYPE_CAPTION",
    "ELEMENT_TYPE_EQUATION",
    "ELEMENT_TYPE_IMAGE",
    "ELEMENT_TYPE_TABLE",
    "ELEMENT_TYPE_TEXT",
    "SUPPORTED_ELEMENT_TYPES",
    "MultimodalElement",
    "from_caption",
    "from_equation",
    "from_image",
    "from_table",
    "from_text_chunk",
    "to_multimodal_element",
    "attach_caption_matches",
    # Multimodal Embedding
    "BaseVisualEmbedder",
    "SentenceTransformerVisualEmbedder",
    "MultimodalEmbeddedElement",
    "MultimodalEmbeddingResult",
    "BaseMultimodalEmbedder",
    "MultimodalEmbedder",
    "get_multimodal_embedder",
    "reset_multimodal_embedder",
    "embed_multimodal_elements",
    # Collection & Assembly
    "MultimodalDocument",
    "MultimodalCollection",
    "assemble_multimodal_document",
    "assemble_multimodal_collection",
]



