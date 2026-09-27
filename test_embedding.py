from embedding import get_embedding

text = "Python Flask machine learning"

embedding = get_embedding(text)

print("Embedding generated successfully")
print("Embedding size:", len(embedding))