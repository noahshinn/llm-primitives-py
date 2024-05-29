from llm_primitives.model import OpenAIModel
from pydantic import BaseModel

model = OpenAIModel(model="gpt-4o")


class Address(BaseModel):
    street: str
    number: int


res = model.parse("I live at 123 main st", Address)
print(f"Street: {res.street}, Number: {res.number}")
