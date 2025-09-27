import asyncio
import os

from agentscope.embedding import FileEmbeddingCache, DashScopeTextEmbedding


async def example_embedding_cache() -> None:
    """Demonstrate embedding with cache functionality."""
    # Example texts
    texts = [
        "What is the capital of France?",
        "Paris is the capital city of France.",
    ]

    # Create a temporary directory for cache demonstration
    # In real applications, you might want to use a persistent directory
    cache_dir = tempfile.mkdtemp(prefix="embedding_cache_")
    print(f"Using cache directory: {cache_dir}")

    # Initialize the embedding model with cache
    # We limit the cache to 100 files and 10MB for demonstration purposes
    embedder = DashScopeTextEmbedding(
        model_name="text-embedding-v3",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        embedding_cache=FileEmbeddingCache(
            cache_dir=cache_dir,
            max_file_number=100,
            max_cache_size=10,  # Maximum cache size in MB
        ),
    )

    # First call - will fetch from API and store in cache
    print("\n=== First API Call (No Cache Hit) ===")
    start_time = asyncio.get_event_loop().time()
    response1 = await embedder(texts)
    elapsed_time1 = asyncio.get_event_loop().time() - start_time
    print(f"Source: {response1.source}")  # Should be 'api'
    print(f"Time taken: {elapsed_time1:.4f} seconds")
    print(f"Tokens used: {response1.usage.tokens}")

    # Second call with the same texts - should use cache
    print("\n=== Second API Call (Cache Hit Expected) ===")
    start_time = asyncio.get_event_loop().time()
    response2 = await embedder(texts)
    elapsed_time2 = asyncio.get_event_loop().time() - start_time
    print(f"Source: {response2.source}")  # Should be 'cache'
    print(f"Time taken: {elapsed_time2:.4f} seconds")
    print(
        f"Tokens used: {response2.usage.tokens}",
    )  # Should be 0 for cached results
    print(
        f"Speed improvement: {elapsed_time1 / elapsed_time2:.1f}x faster with cache",
    )


asyncio.run(example_embedding_cache())
