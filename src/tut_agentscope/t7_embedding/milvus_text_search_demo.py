import warnings

from loguru import logger
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus.milvus_client import IndexParams
from sentence_transformers import SentenceTransformer

# 忽略警告
warnings.filterwarnings('ignore')


def get_text_embedding(text: str, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
    """
    加载模型并获取单个文本的嵌入向量。

    注意：每次调用此函数都会加载一次模型，效率较低。

    :param text: 要转换的文本字符串。
    :param model_name: 要使用的 SentenceTransformer 模型名称。
    :return: 一个 NumPy 数组，代表文本的嵌入向量。
    """
    if not text or not isinstance(text, str):
        raise ValueError("输入必须是一个非空字符串。")

    print(f"正在加载模型: {model_name}...")
    model = SentenceTransformer(model_name)
    print("模型加载完成，正在生成嵌入...")

    return model.encode(text)


try:
    # 连接Milvus数据库
    # client = MilvusClient(uri="./demo.db")
    client = MilvusClient(uri="http://localhost:19530")

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]

    # 定义集合模式
    schema = CollectionSchema(fields=fields, description="Sample collection for storing embeddings")
    collection_name = "embedding_collection"

    # 如果集合存在则删除
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # 创建集合
    client.create_collection(collection_name=collection_name, schema=schema)

    # 定义索引参数
    index_params = IndexParams()
    index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})

    # 创建索引并加载集合
    client.create_index(collection_name=collection_name, index_params=index_params)
    client.load_collection(collection_name)

    # 初始化OpenAI客户端

    # 文本数据
    texts = [
        "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买",
        "价格太贵了，质量一般般，不推荐",
        "这衣服尺码正好，颜色也好看，穿起来很舒适",
        "衣服款式设计太丑了，做工也很粗糙",
        "物流很快，客服态度也很好，下次还会再来",
        "鞋子尺码偏小，而且材质太硬，不喜欢"
    ]

    # 插入数据
    for text in texts:
        # 获取文本嵌入
        embedding_data = get_text_embedding(text)
        # 插入数据到Milvus
        data = [{"embedding": embedding_data, "text": text}]
        client.insert(collection_name=collection_name, data=data)
        print(f"{text} 插入完成")

    # 查询数据
    completion_query = get_text_embedding("衣服的质量杠杠")
    embedding_data_query = completion_query

    # 搜索相似文本
    search_result = client.search(collection_name=collection_name, data=[embedding_data_query], limit=5, output_fields=['text'])

    # 打印搜索结果
    print("\n检索结果如下：")
    for hit in search_result[0]:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}")
        print(f"文本: {hit['entity'].get('text')}")

except Exception as e:
    # 异常处理
    logger.exception("sss", e)
    print(f"错误信息: {e}")
    print("请参考错误码表")
