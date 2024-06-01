import json
from pydantic import Field
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from llm_primitives.model import T, PartialObj


def force_json_prompt(text: str) -> str:
    return f"{text}\n\nValid JSON:"


def display_choices(choices: List[str]) -> Tuple[str, Dict[str, int]]:
    choice_displays = []
    decode_map = {}
    for i, choice in enumerate(choices):
        label = index_to_alpha(i)
        choice_display = f"{label}. {choice}"
        choice_displays.append(choice_display)
        decode_map[label] = i
    return "\n".join(choice_displays), decode_map


def index_to_alpha(index: int) -> str:
    alpha = ""
    while index >= 0:
        alpha = chr(index % 26 + ord("A")) + alpha
        index = index // 26 - 1
    return alpha


def type_to_json_schema_string(typ: Type[T]) -> str:
    json_schema = typ.model_json_schema()
    return json.dumps(json_schema, indent=4)


def optionalize_type(typ: Type[T]) -> Type[T]:
    class OptionalModel(typ): ...

    new_fields = {}
    for name, field in OptionalModel.model_fields.items():
        new_fields[name] = Field(default=None, annotation=Optional[field.annotation])
    OptionalModel.model_fields = new_fields
    return OptionalModel


def json_response_to_obj_or_partial_obj(
    response: Dict[str, Any], typ: Type[T]
) -> Union[T, PartialObj]:
    required_field_names = [
        name for name, field in typ.model_fields.items() if field.is_required()
    ]
    for name in required_field_names:
        if name not in response.keys() or response[name] is None:
            return response
    return typ.model_validate(response)
