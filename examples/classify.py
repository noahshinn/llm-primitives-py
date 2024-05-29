from llm_primitives import OpenAIModel

model = OpenAIModel(model="gpt-4o")
res = model.classify(
    "Determine the sentiment of the text",
    "I love this product",
    ["positive", "negative", "neutral"],
)
print(res)
