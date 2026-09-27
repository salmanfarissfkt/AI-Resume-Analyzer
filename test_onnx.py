from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np


# Load tokenizer
tokenizer = Tokenizer.from_file(
    "models/all-MiniLM-L6-v2/tokenizer.json"
)

tokenizer.enable_truncation(max_length=256)


# Two sentences to compare
texts = [
    "Python Flask machine learning",
    "Python developer with Flask and machine learning experience"
]


# Convert text into tokens
encodings = tokenizer.encode_batch(texts)

# Create model inputs
inputs = {
    "input_ids": np.array(
        [encoding.ids for encoding in encodings],
        dtype=np.int64
    ),

    "attention_mask": np.array(
        [encoding.attention_mask for encoding in encodings],
        dtype=np.int64
    ),

    "token_type_ids": np.array(
        [encoding.type_ids for encoding in encodings],
        dtype=np.int64
    )
}


# Load ONNX model
session = ort.InferenceSession(
    "models/all-MiniLM-L6-v2/model.onnx"
)


# Generate embeddings
output = session.run(
    ["sentence_embedding"],
    inputs
)[0]


print("Embedding shape:", output.shape)


# Get the two embeddings
embedding_1 = output[0]
embedding_2 = output[1]


# Calculate cosine similarity
similarity = np.dot(embedding_1, embedding_2) / (
    np.linalg.norm(embedding_1) *
    np.linalg.norm(embedding_2)
)


print("Cosine similarity:", round(float(similarity), 4))
print("Similarity percentage:", round(float(similarity) * 100, 2), "%")