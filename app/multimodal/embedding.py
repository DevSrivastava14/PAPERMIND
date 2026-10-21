"""Modality-aware multimodal embedding layer for scientific paper elements.

This module provides a unified embedding architecture that routes each MultimodalElement
to its modality-appropriate encoder:
- Text-like elements (text, captions, tables, equations) -> semantic text embedder (BaseEmbedder)
- Visual elements (images) -> local visual embedder (BaseVisualEmbedder / CLIP)

Both streams produce embeddings associated 1:1 with their original MultimodalElement,
preserving exact provenance, element IDs, and metadata.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from app.embeddings.base import BaseEmbedder
from app.multimodal.representation import ElementType, MultimodalElement


# ---------------------------------------------------------------------------
# Result Data Structures
# ---------------------------------------------------------------------------


@dataclass
class MultimodalEmbeddedElement:
    """Represents an embedded MultimodalElement paired with its vector and provenance metadata.

    Attributes:
        element_id: Unique deterministic identifier of the originating element.
        element_type: Modality type (text, image, caption, table, equation).
        source_filename: Name of the originating PDF / document file.
        page_number: 1-based page number where the element appeared.
        vector: 1D float32 NumPy vector embedding.
        element: The original MultimodalElement instance.
        metadata: Modality-specific and embedding metadata dictionary.
    """

    element_id: str
    element_type: ElementType
    source_filename: str
    page_number: int
    vector: np.ndarray
    element: MultimodalElement
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.vector, np.ndarray):
            raise TypeError(
                f"vector must be a NumPy ndarray, got {type(self.vector).__name__}"
            )
        if self.vector.ndim != 1:
            raise ValueError(
                f"vector must be 1-dimensional, got {self.vector.ndim}D with shape {self.vector.shape}"
            )
        if not np.all(np.isfinite(self.vector)):
            raise ValueError("vector contains non-finite values (NaN or Inf)")

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vector."""
        return int(self.vector.shape[0])

    @property
    def is_text(self) -> bool:
        return self.element_type == ElementType.TEXT

    @property
    def is_image(self) -> bool:
        return self.element_type == ElementType.IMAGE

    @property
    def is_caption(self) -> bool:
        return self.element_type == ElementType.CAPTION

    @property
    def is_table(self) -> bool:
        return self.element_type == ElementType.TABLE

    @property
    def is_equation(self) -> bool:
        return self.element_type == ElementType.EQUATION


@dataclass
class MultimodalEmbeddingResult:
    """Holds batch multimodal embeddings while preserving 1:1 mapping and provenance.

    Attributes:
        elements: Original list of MultimodalElement inputs in their given order.
        embedded_elements: List of MultimodalEmbeddedElement outputs matching 1:1.
    """

    elements: List[MultimodalElement]
    embedded_elements: List[MultimodalEmbeddedElement]

    def __len__(self) -> int:
        return len(self.embedded_elements)

    def __iter__(self):
        return iter(self.embedded_elements)

    def __getitem__(self, index: int) -> MultimodalEmbeddedElement:
        return self.embedded_elements[index]

    def get_by_element_id(
        self, element_id: str
    ) -> Optional[MultimodalEmbeddedElement]:
        """Returns the MultimodalEmbeddedElement for a given element_id, or None."""
        for ee in self.embedded_elements:
            if ee.element_id == element_id:
                return ee
        return None

    def get_vector_by_id(self, element_id: str) -> Optional[np.ndarray]:
        """Returns the 1D embedding vector for a given element_id, or None."""
        ee = self.get_by_element_id(element_id)
        return ee.vector if ee is not None else None

    @property
    def element_ids(self) -> List[str]:
        """Ordered list of element IDs in this result."""
        return [ee.element_id for ee in self.embedded_elements]

    @property
    def element_types(self) -> List[ElementType]:
        """Ordered list of element types in this result."""
        return [ee.element_type for ee in self.embedded_elements]

    @property
    def vectors(self) -> List[np.ndarray]:
        """List of 1D NumPy vectors corresponding 1:1 to embedded elements."""
        return [ee.vector for ee in self.embedded_elements]

    def get_vectors_by_type(
        self, element_type: Union[ElementType, str]
    ) -> np.ndarray:
        """Returns a 2D float32 NumPy array of vectors filtered by element type."""
        target = (
            ElementType(element_type)
            if isinstance(element_type, str)
            else element_type
        )
        filtered = [
            ee.vector for ee in self.embedded_elements if ee.element_type == target
        ]
        if not filtered:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(filtered).astype(np.float32)

    def to_numpy_matrix(self) -> np.ndarray:
        """Stacks all vectors into a 2D array if all vectors have uniform dimensionality.

        Raises:
            ValueError: If the result is empty or contains mixed vector dimensions.
        """
        if not self.embedded_elements:
            return np.empty((0, 0), dtype=np.float32)

        dims = {ee.dimension for ee in self.embedded_elements}
        if len(dims) > 1:
            raise ValueError(
                f"Cannot stack mixed-modality vectors with varying dimensions: {sorted(dims)}. "
                "Use .get_vectors_by_type() or iterate over .embedded_elements instead."
            )
        return np.vstack(self.vectors).astype(np.float32)


# ---------------------------------------------------------------------------
# Visual Embedder Interface and Implementation
# ---------------------------------------------------------------------------


class BaseVisualEmbedder(ABC):
    """Abstract base class defining the interface for visual embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the visual embedding vectors."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier or name of the visual embedding model."""
        pass

    @abstractmethod
    def embed_images(
        self,
        images: List[Union[bytes, str, Path, Any]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embeds raw images (bytes, file paths, or PIL Images) into a 2D float32 array."""
        pass

    def embed_text_queries(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embeds text into the same space as images when the model supports it."""
        raise NotImplementedError(
            f"Visual embedder '{self.model_name}' does not support text queries"
        )


class SentenceTransformerVisualEmbedder(BaseVisualEmbedder):
    """Local pretrained visual embedder leveraging sentence-transformers (e.g. CLIP)."""

    DEFAULT_MODEL_NAME: str = "sentence-transformers/clip-ViT-B-32"
    DEFAULT_DIMENSION: int = 512

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        device: Optional[str] = None,
        model: Optional[Any] = None,
        dimension: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._model_name = model_name
        self.default_batch_size = batch_size
        self.device = device
        self._dimension = dimension
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        if self._model is not None and hasattr(
            self._model, "get_sentence_embedding_dimension"
        ):
            dim = self._model.get_sentence_embedding_dimension()
            if dim is not None:
                self._dimension = int(dim)
                return self._dimension
        return self.DEFAULT_DIMENSION

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self.device)
        return self._model

    def embed_images(
        self,
        images: List[Union[bytes, str, Path, Any]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embeds a list of images into a 2D float32 NumPy array of shape (N, dimension)."""
        effective_batch_size = (
            batch_size if batch_size is not None else self.default_batch_size
        )
        if effective_batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {effective_batch_size}"
            )

        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for visual embeddings. Install it via `pip install Pillow`."
            ) from exc

        # Preprocess each input into a valid PIL RGB Image
        pil_images = []
        for idx, item in enumerate(images):
            if isinstance(item, bytes):
                if len(item) == 0:
                    raise ValueError(f"Image bytes at index {idx} cannot be empty")
                try:
                    pil_img = Image.open(io.BytesIO(item)).convert("RGB")
                except Exception as exc:
                    raise ValueError(
                        f"Failed to decode image bytes at index {idx}: {exc}"
                    ) from exc
            elif isinstance(item, (str, Path)):
                str_path = str(item)
                if not str_path.strip():
                    raise ValueError(f"Image path at index {idx} cannot be empty")
                try:
                    pil_img = Image.open(str_path).convert("RGB")
                except Exception as exc:
                    raise ValueError(
                        f"Failed to open image file '{str_path}' at index {idx}: {exc}"
                    ) from exc
            elif hasattr(item, "convert"):  # Already a PIL Image
                pil_img = item.convert("RGB")
            else:
                raise TypeError(
                    f"Unsupported image input type at index {idx}: {type(item).__name__}"
                )
            pil_images.append(pil_img)

        model = self._get_model()
        batches: List[np.ndarray] = []
        for i in range(0, len(pil_images), effective_batch_size):
            batch_items = pil_images[i : i + effective_batch_size]
            raw_output = model.encode(
                batch_items,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            if hasattr(raw_output, "detach"):
                raw_output = raw_output.detach().cpu().numpy()

            batch_array = np.asarray(raw_output, dtype=np.float32)
            if batch_array.ndim == 1:
                batch_array = np.expand_dims(batch_array, axis=0)

            batches.append(batch_array)

        vectors = np.vstack(batches)
        self._validate_vectors(vectors, expected_count=len(images))
        return vectors

    def embed_text_queries(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        effective_batch_size = (
            batch_size if batch_size is not None else self.default_batch_size
        )
        if effective_batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {effective_batch_size}"
            )
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        if "clip" not in self.model_name.lower():
            raise NotImplementedError(
                f"Visual model '{self.model_name}' is not verified as a shared-space CLIP model"
            )

        model = self._get_model()
        if not callable(getattr(model, "encode", None)):
            raise NotImplementedError(
                f"Visual model '{self.model_name}' does not support text queries"
            )

        raw_output = model.encode(
            texts,
            batch_size=effective_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if hasattr(raw_output, "detach"):
            raw_output = raw_output.detach().cpu().numpy()
        vectors = np.asarray(raw_output, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = np.expand_dims(vectors, axis=0)
        self._validate_vectors(vectors, expected_count=len(texts))
        return vectors

    def _validate_vectors(self, vectors: np.ndarray, expected_count: int) -> None:
        """Validates that vectors are 2D NumPy float32 arrays, match expected dimensions, and are finite."""
        if not isinstance(vectors, np.ndarray):
            raise TypeError(
                f"Vectors must be a NumPy ndarray, got {type(vectors).__name__}"
            )

        if vectors.ndim != 2:
            raise ValueError(
                f"Vectors must be 2-dimensional (shape: [num_vectors, dimension]), "
                f"got {vectors.ndim}D array with shape {vectors.shape}"
            )

        if vectors.shape[0] != expected_count:
            raise ValueError(
                f"Vector count mismatch: expected {expected_count} vectors, got {vectors.shape[0]}"
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected dimension {self.dimension}, got {vectors.shape[1]}"
            )

        if not np.all(np.isfinite(vectors)):
            raise ValueError("Embedding vectors contain non-finite values (NaN or Inf)")


# ---------------------------------------------------------------------------
# Multimodal Embedder Base Interface & Orchestrator
# ---------------------------------------------------------------------------


class BaseMultimodalEmbedder(ABC):
    """Abstract base class defining the unified interface for multimodal embedding."""

    @property
    @abstractmethod
    def text_dimension(self) -> int:
        """Dimensionality of vectors for text-like elements."""
        pass

    @property
    @abstractmethod
    def visual_dimension(self) -> int:
        """Dimensionality of vectors for visual elements."""
        pass

    @property
    @abstractmethod
    def text_model_name(self) -> str:
        """Identifier or name of the text embedding model."""
        pass

    @property
    @abstractmethod
    def visual_model_name(self) -> str:
        """Identifier or name of the visual embedding model."""
        pass

    @abstractmethod
    def embed_element(
        self,
        element: MultimodalElement,
    ) -> MultimodalEmbeddedElement:
        """Embeds a single MultimodalElement."""
        pass

    @abstractmethod
    def embed_elements(
        self,
        elements: List[MultimodalElement],
        batch_size: Optional[int] = None,
    ) -> MultimodalEmbeddingResult:
        """Embeds a batch of MultimodalElements preserving 1:1 mapping and provenance."""
        pass


class MultimodalEmbedder(BaseMultimodalEmbedder):
    """Unified multimodal embedder routing elements to modality-appropriate models.

    - Text, Caption, Table, Equation -> semantic text embedder (BaseEmbedder)
    - Image -> local visual embedder (BaseVisualEmbedder)
    """

    def __init__(
        self,
        text_embedder: Optional[BaseEmbedder] = None,
        visual_embedder: Optional[BaseVisualEmbedder] = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.default_batch_size = batch_size

        self._text_embedder = text_embedder
        self._visual_embedder = visual_embedder

    @property
    def text_embedder(self) -> BaseEmbedder:
        if self._text_embedder is None:
            from app.embeddings.sentence_transformer_embedder import get_embedder

            self._text_embedder = get_embedder()
        return self._text_embedder

    @property
    def visual_embedder(self) -> BaseVisualEmbedder:
        if self._visual_embedder is None:
            self._visual_embedder = SentenceTransformerVisualEmbedder()
        return self._visual_embedder

    @property
    def text_dimension(self) -> int:
        return self.text_embedder.dimension

    @property
    def visual_dimension(self) -> int:
        return self.visual_embedder.dimension

    @property
    def text_model_name(self) -> str:
        return self.text_embedder.model_name

    @property
    def visual_model_name(self) -> str:
        return self.visual_embedder.model_name

    def embed_element(
        self, element: MultimodalElement
    ) -> MultimodalEmbeddedElement:
        """Embeds a single MultimodalElement."""
        if element is None:
            raise ValueError("element cannot be None")
        result = self.embed_elements([element])
        return result.embedded_elements[0]

    def embed_elements(
        self,
        elements: List[MultimodalElement],
        batch_size: Optional[int] = None,
    ) -> MultimodalEmbeddingResult:
        """Embeds a list of MultimodalElements preserving 1:1 mapping and provenance.

        Elements are internally partitioned by modality (text-like vs visual),
        encoded using the modality-appropriate model, and reconstructed in the exact
        original input order.
        """
        if not elements:
            return MultimodalEmbeddingResult(elements=[], embedded_elements=[])

        effective_batch_size = (
            batch_size if batch_size is not None else self.default_batch_size
        )
        if effective_batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {effective_batch_size}"
            )

        text_indices: List[int] = []
        text_inputs: List[str] = []

        image_indices: List[int] = []
        image_inputs: List[Union[bytes, str, Any]] = []

        for idx, el in enumerate(elements):
            if not isinstance(el, MultimodalElement):
                raise TypeError(
                    f"Element at index {idx} must be a MultimodalElement, got {type(el).__name__}"
                )

            if el.element_type in (
                ElementType.TEXT,
                ElementType.CAPTION,
                ElementType.TABLE,
                ElementType.EQUATION,
            ):
                # Text-like elements: extract text payload
                content_text = el.text if el.text else str(el.content)
                text_indices.append(idx)
                text_inputs.append(content_text)
            elif el.element_type == ElementType.IMAGE:
                # Visual elements: extract binary payload or path
                image_payload: Union[bytes, str]
                if el.image_data:
                    image_payload = el.image_data
                elif el.image_path:
                    image_payload = el.image_path
                elif isinstance(el.content, (bytes, str)):
                    image_payload = el.content
                else:
                    raise ValueError(
                        f"Image element '{el.element_id}' has neither image_data nor image_path"
                    )
                image_indices.append(idx)
                image_inputs.append(image_payload)
            else:
                raise ValueError(
                    f"Unsupported element_type '{el.element_type}' for element '{el.element_id}'"
                )

        # Encode text-like elements
        text_vectors: Optional[np.ndarray] = None
        if text_inputs:
            text_vectors = self.text_embedder.embed_texts(
                text_inputs, batch_size=effective_batch_size
            )

        # Encode visual elements
        image_vectors: Optional[np.ndarray] = None
        if image_inputs:
            image_vectors = self.visual_embedder.embed_images(
                image_inputs, batch_size=effective_batch_size
            )

        # Restore original ordering and construct MultimodalEmbeddedElement items
        embedded_elements: List[Optional[MultimodalEmbeddedElement]] = [
            None
        ] * len(elements)

        if text_vectors is not None:
            for local_idx, orig_idx in enumerate(text_indices):
                vec = text_vectors[local_idx]
                el = elements[orig_idx]
                el.embedding = vec
                metadata = dict(el.metadata)
                metadata["embedding_model"] = self.text_model_name
                metadata["modality_stream"] = "text"
                embedded_elements[orig_idx] = MultimodalEmbeddedElement(
                    element_id=el.element_id,
                    element_type=el.element_type,
                    source_filename=el.source_filename,
                    page_number=el.page_number,
                    vector=vec,
                    element=el,
                    metadata=metadata,
                )

        if image_vectors is not None:
            for local_idx, orig_idx in enumerate(image_indices):
                vec = image_vectors[local_idx]
                el = elements[orig_idx]
                el.embedding = vec
                metadata = dict(el.metadata)
                metadata["embedding_model"] = self.visual_model_name
                metadata["modality_stream"] = "visual"
                embedded_elements[orig_idx] = MultimodalEmbeddedElement(
                    element_id=el.element_id,
                    element_type=el.element_type,
                    source_filename=el.source_filename,
                    page_number=el.page_number,
                    vector=vec,
                    element=el,
                    metadata=metadata,
                )

        final_embedded: List[MultimodalEmbeddedElement] = [
            ee for ee in embedded_elements if ee is not None
        ]
        return MultimodalEmbeddingResult(
            elements=list(elements),
            embedded_elements=final_embedded,
        )


# ---------------------------------------------------------------------------
# Global Singleton and Convenience Functions
# ---------------------------------------------------------------------------

_DEFAULT_MULTIMODAL_EMBEDDER: Optional[MultimodalEmbedder] = None


def get_multimodal_embedder(
    text_embedder: Optional[BaseEmbedder] = None,
    visual_embedder: Optional[BaseVisualEmbedder] = None,
) -> MultimodalEmbedder:
    """Returns a shared or newly configured MultimodalEmbedder instance."""
    global _DEFAULT_MULTIMODAL_EMBEDDER
    if (
        _DEFAULT_MULTIMODAL_EMBEDDER is None
        or text_embedder is not None
        or visual_embedder is not None
    ):
        _DEFAULT_MULTIMODAL_EMBEDDER = MultimodalEmbedder(
            text_embedder=text_embedder,
            visual_embedder=visual_embedder,
        )
    return _DEFAULT_MULTIMODAL_EMBEDDER


def reset_multimodal_embedder() -> None:
    """Resets the shared MultimodalEmbedder instance (useful for clean test teardowns)."""
    global _DEFAULT_MULTIMODAL_EMBEDDER
    _DEFAULT_MULTIMODAL_EMBEDDER = None


def embed_multimodal_elements(
    elements: List[MultimodalElement],
    embedder: Optional[BaseMultimodalEmbedder] = None,
    batch_size: Optional[int] = None,
) -> MultimodalEmbeddingResult:
    """Convenience function to embed a list of MultimodalElement objects."""
    if embedder is None:
        embedder = get_multimodal_embedder()
    return embedder.embed_elements(elements, batch_size=batch_size)
