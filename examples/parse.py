from llm_primitives import OpenAIModel
from pydantic import BaseModel


class Address(BaseModel):
    street: str
    number: int


model = OpenAIModel(model="gpt-4o")
res = model.parse("I live at 123 main st", Address)
print(f"Street: {res.street}, Number: {res.number}")
