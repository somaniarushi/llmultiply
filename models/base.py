from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def generate(
        self, 
        prompt: str,
        max_tokens_to_generate: int,
        temperature: float,
        top_k: int, 
    ) -> str:
        raise NotImplementedError(f"generate method not implemented for {self.__class__.__name__}")

