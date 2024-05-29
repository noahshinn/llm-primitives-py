from llm_primitives import OpenAIModel

model = OpenAIModel(model="gpt-4o")
res = model.binary_classify(
    "Determine if the product review is positive", "I love this product"
)

print(res)
