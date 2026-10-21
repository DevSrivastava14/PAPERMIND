"""Retrieval evaluation runner across four benchmark retrieval configurations.

Executes and benchmarks:
1. Dense FAISS retrieval (384-d semantic search)
2. BM25 retrieval (lexical/sparse search)
3. Hybrid Dense + BM25 -> RRF (Reciprocal Rank Fusion)
4. Hybrid Dense + BM25 -> RRF -> Cross-Encoder (neural reranking)

Preserves the original MultimodalSearchResult objects and full provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from app.embeddings.base import BaseEmbedder
from app.evaluation.models import EvaluationDataset, EvaluationQuestion
from app.indexing.bm25_index import BM25Index, build_bm25_index
from app.indexing.multimodal_faiss_index import (
    MultimodalFAISSIndex,
    build_multimodal_index,
)
from app.indexing.multimodal_models import MultimodalSearchResult
from app.multimodal.collection import MultimodalDocument
from app.multimodal.embedding import BaseVisualEmbedder, MultimodalEmbedder
from app.reranking.base import BaseReranker
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.multimodal_pipeline import (
    MultimodalRetrievalPipeline,
    build_multimodal_pipeline,
)
from app.retrieval.rrf import DEFAULT_RRF_K


class RetrievalConfiguration(str, Enum):
    """The four benchmark retrieval configurations."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANKED = "hybrid_reranked"

    @classmethod
    def from_str(cls, value: str) -> RetrievalConfiguration:
        """Parse configuration string with convenient aliases."""
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "dense": cls.DENSE,
            "faiss": cls.DENSE,
            "vector": cls.DENSE,
            "bm25": cls.BM25,
            "sparse": cls.BM25,
            "lexical": cls.BM25,
            "hybrid": cls.HYBRID_RRF,
            "hybrid_rrf": cls.HYBRID_RRF,
            "rrf": cls.HYBRID_RRF,
            "reranked": cls.HYBRID_RERANKED,
            "hybrid_reranked": cls.HYBRID_RERANKED,
            "cross_encoder": cls.HYBRID_RERANKED,
            "rerank": cls.HYBRID_RERANKED,
        }
        if normalized in aliases:
            return aliases[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unknown retrieval configuration: '{value}'. Expected one of: "
            f"{[m.value for m in cls]}"
        )


ALL_RETRIEVAL_CONFIGURATIONS: List[RetrievalConfiguration] = [
    RetrievalConfiguration.DENSE,
    RetrievalConfiguration.BM25,
    RetrievalConfiguration.HYBRID_RRF,
    RetrievalConfiguration.HYBRID_RERANKED,
]


@dataclass
class QuestionRetrievalRun:
    """Results and execution metadata of running one EvaluationQuestion across retrieval configurations.

    Attributes:
        question: The input EvaluationQuestion.
        results: Mapping from configuration name to ranked MultimodalSearchResult list.
        latencies: Mapping from configuration name to execution duration in seconds.
    """

    question: EvaluationQuestion
    results: Dict[str, List[MultimodalSearchResult]] = field(default_factory=dict)
    latencies: Dict[str, float] = field(default_factory=dict)

    @property
    def dense_results(self) -> List[MultimodalSearchResult]:
        """Results from Dense FAISS retrieval."""
        return self.results.get(RetrievalConfiguration.DENSE.value, [])

    @property
    def bm25_results(self) -> List[MultimodalSearchResult]:
        """Results from BM25 sparse retrieval."""
        return self.results.get(RetrievalConfiguration.BM25.value, [])

    @property
    def hybrid_rrf_results(self) -> List[MultimodalSearchResult]:
        """Results from Hybrid Dense + BM25 -> RRF."""
        return self.results.get(RetrievalConfiguration.HYBRID_RRF.value, [])

    @property
    def hybrid_reranked_results(self) -> List[MultimodalSearchResult]:
        """Results from Hybrid Dense + BM25 -> RRF -> Cross-Encoder."""
        return self.results.get(RetrievalConfiguration.HYBRID_RERANKED.value, [])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize run results summary to dictionary."""
        return {
            "question_id": self.question.id,
            "query": self.question.query,
            "question_type": self.question.question_type.value,
            "latencies": dict(self.latencies),
            "results": {
                config_name: [
                    {
                        "element_id": r.element_id,
                        "element_type": r.element_type.value if hasattr(r.element_type, "value") else str(r.element_type),
                        "page_number": r.page_number,
                        "score": r.score,
                        "source_filename": r.source_filename,
                        "content": str(r.content) if r.content is not None else "",
                        "content_preview": (str(r.content)[:120] + "...") if r.content and len(str(r.content)) > 120 else str(r.content or ""),
                    }

                    for r in res_list
                ]
                for config_name, res_list in self.results.items()
            },
        }


def create_evaluation_pipeline(
    document: MultimodalDocument,
    text_embedder: Optional[BaseEmbedder] = None,
    visual_embedder: Optional[BaseVisualEmbedder] = None,
    reranker: Optional[BaseReranker] = None,
    default_top_k: int = 5,
) -> MultimodalRetrievalPipeline:
    """Builds a complete MultimodalRetrievalPipeline ready for all four retrieval configurations.

    Instantiates:
    1. MultimodalFAISSIndex (Dense)
    2. BM25Index (Sparse)
    3. CrossEncoderReranker (Neural reranker)
    """
    if text_embedder is not None or visual_embedder is not None:
        embedder = MultimodalEmbedder(
            text_embedder=text_embedder,
            visual_embedder=visual_embedder,
        )
        faiss_index = build_multimodal_index(document, embedder=embedder)
    else:
        faiss_index = build_multimodal_index(document)

    bm25_index = build_bm25_index(document)
    active_reranker = reranker if reranker is not None else CrossEncoderReranker()

    return build_multimodal_pipeline(
        index=faiss_index,
        text_embedder=text_embedder,
        visual_embedder=visual_embedder,
        bm25_index=bm25_index,
        reranker=active_reranker,
        default_top_k=default_top_k,
    )


class RetrievalEvaluationRunner:
    """Orchestrates retrieval evaluation runs across the four benchmark configurations.

    Executes:
    1. Dense FAISS: `pipeline.retrieve_text()`
    2. BM25: `pipeline.retrieve_sparse()`
    3. Hybrid RRF: `pipeline.hybrid_retrieve_text()`
    4. Hybrid Reranked: `pipeline.hybrid_retrieve_reranked()`
    """

    def __init__(
        self,
        pipeline: MultimodalRetrievalPipeline,
        default_top_k: int = 5,
        default_rrf_k: int = DEFAULT_RRF_K,
        configurations: Optional[Sequence[Union[RetrievalConfiguration, str]]] = None,
    ) -> None:
        if not isinstance(pipeline, MultimodalRetrievalPipeline):
            raise TypeError(
                f"pipeline must be an instance of MultimodalRetrievalPipeline, got {type(pipeline).__name__}"
            )
        if default_top_k <= 0:
            raise ValueError(f"default_top_k must be a positive integer, got {default_top_k}")
        if default_rrf_k <= 0:
            raise ValueError(f"default_rrf_k must be a positive integer, got {default_rrf_k}")

        self.pipeline = pipeline
        self.default_top_k = default_top_k
        self.default_rrf_k = default_rrf_k

        if configurations is not None:
            self.configurations = [
                RetrievalConfiguration.from_str(c) if isinstance(c, str) else c
                for c in configurations
            ]
        else:
            self.configurations = list(ALL_RETRIEVAL_CONFIGURATIONS)

    def run_question(
        self,
        question: EvaluationQuestion,
        top_k: Optional[int] = None,
        configurations: Optional[Sequence[Union[RetrievalConfiguration, str]]] = None,
        rrf_k: Optional[int] = None,
    ) -> QuestionRetrievalRun:
        """Executes the specified retrieval configurations for a single EvaluationQuestion.

        Args:
            question: EvaluationQuestion instance from the dataset.
            top_k: Optional limit on retrieved results per configuration (defaults to self.default_top_k).
            configurations: Optional override of configurations to execute.
            rrf_k: Optional smoothing constant for RRF fusion.

        Returns:
            QuestionRetrievalRun containing results and timing for each configuration.
        """
        if not isinstance(question, EvaluationQuestion):
            raise TypeError(
                f"question must be an instance of EvaluationQuestion, got {type(question).__name__}"
            )

        effective_k = self.default_top_k if top_k is None else top_k
        if effective_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {effective_k}")

        effective_rrf_k = self.default_rrf_k if rrf_k is None else rrf_k
        if effective_rrf_k <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got {effective_rrf_k}")

        target_configs: List[RetrievalConfiguration]
        if configurations is not None:
            target_configs = [
                RetrievalConfiguration.from_str(c) if isinstance(c, str) else c
                for c in configurations
            ]
        else:
            target_configs = self.configurations

        results: Dict[str, List[MultimodalSearchResult]] = {}
        latencies: Dict[str, float] = {}

        for config in target_configs:
            start_time = time.perf_counter()

            if config == RetrievalConfiguration.DENSE:
                res = self.pipeline.retrieve_text(query=question.query, top_k=effective_k)
            elif config == RetrievalConfiguration.BM25:
                res = self.pipeline.retrieve_sparse(query=question.query, top_k=effective_k)
            elif config == RetrievalConfiguration.HYBRID_RRF:
                res = self.pipeline.hybrid_retrieve_text(
                    query=question.query,
                    top_k=effective_k,
                    rrf_k=effective_rrf_k,
                )
            elif config == RetrievalConfiguration.HYBRID_RERANKED:
                res = self.pipeline.hybrid_retrieve_reranked(
                    query=question.query,
                    top_k=effective_k,
                    rrf_k=effective_rrf_k,
                )
            else:
                raise ValueError(f"Unsupported retrieval configuration: {config}")

            duration = time.perf_counter() - start_time
            results[config.value] = res
            latencies[config.value] = duration

        return QuestionRetrievalRun(
            question=question,
            results=results,
            latencies=latencies,
        )

    def run_dataset(
        self,
        dataset: Union[EvaluationDataset, Sequence[EvaluationQuestion]],
        top_k: Optional[int] = None,
        configurations: Optional[Sequence[Union[RetrievalConfiguration, str]]] = None,
        rrf_k: Optional[int] = None,
    ) -> List[QuestionRetrievalRun]:
        """Executes retrieval configurations across all questions in an EvaluationDataset.

        Args:
            dataset: EvaluationDataset or sequence of EvaluationQuestion objects.
            top_k: Optional limit on retrieved results per configuration.
            configurations: Optional override of configurations to execute.
            rrf_k: Optional smoothing constant for RRF.

        Returns:
            List of QuestionRetrievalRun instances.
        """
        questions = dataset.questions if isinstance(dataset, EvaluationDataset) else dataset
        runs: List[QuestionRetrievalRun] = []
        for q in questions:
            run = self.run_question(
                question=q,
                top_k=top_k,
                configurations=configurations,
                rrf_k=rrf_k,
            )
            runs.append(run)
        return runs
