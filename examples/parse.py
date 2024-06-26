from llm_primitives import OpenAIModel
from pydantic import BaseModel


class Address(BaseModel):
    street: str
    number: int


model = OpenAIModel(model="gpt-4o")
addr = model.parse("I live at 123 main st", Address)
assert isinstance(addr, Address)
print(f"Street: {addr.street}, Number: {addr.number}")
