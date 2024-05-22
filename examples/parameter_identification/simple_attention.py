import numpy as np


# Simple softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def attention(query, keys, values, scale_factor=1.0):
    """
    Simple dot-product attention mechanism.

    Args:
    - query: A vector to compute attention scores (shape: [dim]).
    - keys: A matrix where each row is a key vector (shape: [num_keys, dim]).
    - values: A matrix where each row is a value vector. Typically same as keys (shape: [num_keys, dim]).
    - scale_factor: A scaling factor for the dot product (e.g., sqrt(dim) is commonly used).

    Returns:
    - context: Weighted sum of values based on attention scores.
    """

    # Compute dot product between query and each key
    scores = np.dot(keys, query) / scale_factor

    # Compute attention weights
    attention_weights = softmax(scores)

    # Compute context vector as weighted sum of values
    context = np.sum(values.T * attention_weights, axis=1)

    return context, attention_weights


# Define some keys and values (for simplicity, we'll assume they're the same)
keys_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Define a query
query = np.array([2, 3, 4])

# Compute context and attention weights
context_vector, attention_w = attention(
    query, keys_values, keys_values, scale_factor=np.sqrt(3)
)

print("Attention Weights:", attention_w)
print("Context Vector:", context_vector)
