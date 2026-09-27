from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np


# Load tokenizer
tokenizer = Tokenizer.from_file(
    "models/all-MiniLM-L6-v2/tokenizer.json"
)

tokenizer.enable_truncation(max_length=256)


# Load ONNX model
session = ort.InferenceSession(
    "models/all-MiniLM-L6-v2/model.onnx"
)


def get_embedding(text):

    # Convert text into tokens
    encoding = tokenizer.encode(text)

    # Prepare model inputs
    inputs = {
        "input_ids": np.array(
            [encoding.ids],
            dtype=np.int64
        ),

        "attention_mask": np.array(
            [encoding.attention_mask],
            dtype=np.int64
        ),

        "token_type_ids": np.array(
            [encoding.type_ids],
            dtype=np.int64
        )
    }

    # Generate embedding
    embedding = session.run(
        ["sentence_embedding"],
        inputs
    )[0][0]

    return embedding