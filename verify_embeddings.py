
from app.embeddings import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder()

texts = [
    "We evaluate a neural network on scientific datasets.",
    "The experiment measures model performance on data.",
    "A banana is a tropical fruit.",
]

vectors = embedder.embed_texts(texts)

print("Model:", embedder.model_name)
print("Vector shape:", vectors.shape)
print("Data type:", vectors.dtype)
print("All values finite:", __import__("numpy").isfinite(vectors).all())
print("First vector preview:", vectors[0][:5])

assert vectors.shape == (3, 384)
assert vectors.dtype.name == "float32"
assert __import__("numpy").isfinite(vectors).all()

print("Real embedding verification passed!")
