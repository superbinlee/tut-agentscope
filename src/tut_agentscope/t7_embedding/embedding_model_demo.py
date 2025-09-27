from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embeddings = model.encode(["这是一个测试句子。"])
print(embeddings.shape)
