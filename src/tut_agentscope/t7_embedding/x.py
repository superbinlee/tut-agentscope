from sentence_transformers import SentenceTransformer

# 自动下载并加载模型
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# 示例使用：编码文本
embeddings = model.encode(["这是一个测试句子。"])
print(embeddings)
