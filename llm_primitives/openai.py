import enum
import json
import os
from typing import List, Optional, Dict, Any, Type, Union
from pydantic import BaseModel
from llm_primitives.model import Model, T, PartialObj
from llm_primitives.utils import (
    force_json_prompt,
    display_choices,
    json_response_to_obj_or_partial_obj,
    type_to_json_schema_string,
    optionalize_type,
)


class Role(enum.Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class Message(BaseModel):
    role: Role
    content: str
    obj: Optional[Dict[str, Any]] = None


class OpenAIMessage(BaseModel):
    role: Role
    content: str


class OpenAIModel(Model):
    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self.model = model
        from openai import OpenAI

        api_key = None
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)

    def generate_message(
        self,
        messages: List[Message],
        force_json: bool,
    ) -> Message:
        msgs: List[Dict[str, str]] = []
        for msg in messages:
            if msg.obj is not None:
                content = json.dumps(msg.obj)
            else:
                content = msg.content
            msgs.append({"role": msg.role.value, "content": content})
        res = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=0.0,
            response_format={"type": "json_object" if force_json else "text"},
        )
        content = res.choices[0].message.content
        assert content is not None
        if force_json:
            return Message(role=Role.SYSTEM, content=content, obj=json.loads(content))
        return Message(role=Role.SYSTEM, content=content, obj=None)

    def classify(self, instruction: str, text: str, choices: List[str]) -> int:
        choices_display, decode_map = display_choices(choices)
        input_text = force_json_prompt(
            f"Instruction:\n{instruction}\n\nText:\n{text}\n\nChoices:\n{choices_display}"
        )
        messages = [
            Message(
                role=Role.SYSTEM,
                content='Classify the following text with the provided instruction and choices. To classify, provide the key of the choice:\n{"classification": string}\n\nFor example, if the correct choice is \'Z. description of choice Z\', then provide \'Z\' as the classification as valid JSON:\n{"classification": "Z"}',
            ),
            Message(role=Role.USER, content=input_text),
        ]
        res = self.generate_message(messages, force_json=True)
        if res.obj is None or "classification" not in res.obj:
            raise ValueError(f"Invalid response from model: {res.content}")
        choice = res.obj["classification"]
        if choice not in decode_map:
            raise ValueError(f"Invalid choice: {choice}")
        return decode_map[choice]

    def parse(self, text: str, typ: Type[T]) -> Union[T, PartialObj]:
        optionalized_typ = optionalize_type(typ)
        json_schema_string = type_to_json_schema_string(optionalized_typ)
        input_text = force_json_prompt(
            f"Text:\n{text}\n\nSchema:\n{json_schema_string}"
        )
        messages = [
            Message(
                role=Role.SYSTEM,
                content="Parse the following text with the provided JSON schema.",
            ),
            Message(
                role=Role.USER,
                content=input_text,
            ),
        ]
        res = self.generate_message(messages, force_json=True)
        if res.obj is None:
            return None
        return json_response_to_obj_or_partial_obj(response=res.obj, typ=typ)

    def generate_text(self, instruction: str, text: str) -> str:
        messages = [
            Message(role=Role.SYSTEM, content=instruction),
            Message(role=Role.USER, content=text),
        ]
        return self.generate_message(messages, force_json=False).content

    def score(
        self,
        instruction: str,
        text: str,
        min: Union[int, float],
        max: Union[int, float],
    ) -> Union[int, float]:
        if not (
            (isinstance(min, int) and isinstance(max, int))
            or (isinstance(min, float) and isinstance(max, float))
        ):
            raise ValueError(f"Invalid range types: {type(min)}, {type(max)}")
        typ = type(min)
        if min > max:
            raise ValueError(f"Invalid range: [{min}, {max}]")
        input_text = force_json_prompt(
            f"Instruction:\n{instruction}\n\nText:\n{text}\n\nRange:\n[{min}, {max}]"
        )
        messages = [
            Message(
                role=Role.SYSTEM,
                content=f'Score the following text with the provided instruction and range as a {typ} value as valid JSON:\n{{"score": {typ}}}',
            ),
            Message(role=Role.USER, content=input_text),
        ]
        res = self.generate_message(messages, force_json=True)
        if res.obj is None or "score" not in res.obj:
            raise ValueError(f"Invalid response from model: {res.content}")
        score = res.obj["score"]
        if not isinstance(score, (int, float)):
            raise ValueError(f"Invalid score type: {type(score)}")
        if score < min or score > max:
            raise ValueError(f"Invalid score value: {score}")
        return typ(score)
