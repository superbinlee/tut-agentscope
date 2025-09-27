import asyncio
import os
import tempfile

from agentscope.embedding import DashScopeTextEmbedding, FileEmbeddingCache


async def example_dashscope_embedding() -> None:
    """Example usage of DashScope text embedding."""
    texts = [
        "What is the capital of France?",
        "Paris is the capital city of France.",
    ]

    # Initialize the DashScope text embedding instance
    embedding_model = DashScopeTextEmbedding(
        model_name="text-embedding-v2",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    # Get the embedding from the model
    response = await embedding_model(texts)

    print("The embedding ID: ", response.id)
    print("The embedding create at: ", response.created_at)
    print("The embedding usage: ", response.usage)
    print("The embedding:")
    print(response.embeddings)


asyncio.run(example_dashscope_embedding())
