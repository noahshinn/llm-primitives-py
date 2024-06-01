import abc

from typing import List, TypeVar, Union, Type, Dict, Any
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
PartialObj = Dict[str, Any]


class Model(abc.ABC):
    @abc.abstractmethod
    def classify(self, instruction: str, text: str, choices: List[str]) -> int:
        raise NotImplementedError

    def binary_classify(self, instruction: str, text: str) -> bool:
        return self.classify(instruction, text, ["true", "false"]) == 0

    @abc.abstractmethod
    def parse(self, text: str, typ: Type[T]) -> Union[T, PartialObj]:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_text(self, instruction: str, text: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def score(
        self,
        instruction: str,
        text: str,
        min: Union[int, float],
        max: Union[int, float],
    ) -> Union[int, float]:
        raise NotImplementedError
