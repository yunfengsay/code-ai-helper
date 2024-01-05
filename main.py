from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact")

generator = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

def respond_to_query(query):
    inputs = tokenizer(query, return_tensors="pt")
    res = generator.generate(input_ids=inputs["input_ids"], num_return_sequences=1)
    return tokenizer.decode(res[0], skip_special_tokens=True)

# Example usage:
user_query = input("Enter your query: ")
response = respond_to_query(user_query)
print("AI Response:", response)
