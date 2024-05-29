from llm_primitives import OpenAIModel


model = OpenAIModel(model="gpt-4o")
res = model.score(
    "Score the product review from (1) good to (5) bad", "I love this product", 1, 5
)
print(f"Score: {res}")
