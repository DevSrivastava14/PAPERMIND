from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract base class defining the provider-independent generation interface.

    Concrete implementations may wrap any LLM provider (OpenAI, Anthropic, a local
    model, etc.) without requiring changes to the retrieval or orchestration layers.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier or name of the underlying generation model."""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a response for the given prompt string.

        Args:
            prompt: The fully assembled prompt to send to the model.

        Returns:
            The model's response as a plain string.
        """
        pass
