import asyncio

from agentscope.agent import ReActAgent
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse

# 模型配置参数
model_name = "deepseek-chat"
api_key = "sk-bf45e97095c64d8aa0336a7857563493"
base_url = "https://api.deepseek.com"


async def generate_python(demand: str) -> ToolResponse:
    """Generate Python code based on the demand.

    Args:
        demand (``str``):
            The demand for the Python code.
    """
    # 创建路由智能体
    python_agent = ReActAgent(
        name="PythonAgent",
        sys_prompt="You're a Python expert, your target is to generate Python code based on the demand.",
        model=OpenAIChatModel(api_key=api_key, model_name=model_name, client_args={"base_url": base_url}, stream=False),
        formatter=DeepSeekChatFormatter(),  # 格式化器
        memory=InMemoryMemory(),  # 内存存储
    )
    msg_res = await python_agent(Msg("user", demand, "user"))
    return ToolResponse(content=msg_res.get_content_blocks("text"), )


# Fake some other tool functions for demonstration purposes
async def generate_poem(demand: str) -> ToolResponse:
    """Generate a poem based on the demand.

    Args:
        demand (``str``):
            The demand for the poem.
    """
    pass


async def web_search(query: str) -> ToolResponse:
    """Search the web for the query.

    Args:
        query (``str``):
            The query to search.
    """
    pass


toolkit = Toolkit()
toolkit.register_tool_function(generate_python)
toolkit.register_tool_function(generate_poem)
toolkit.register_tool_function(web_search)

# Initialize the routing agent with the toolkit
router_implicit = ReActAgent(
    name="Router",
    sys_prompt="You're a routing agent. Your target is to route the user query to the right follow-up task.",
    model=OpenAIChatModel(api_key=api_key, model_name=model_name, client_args={"base_url": base_url}, stream=False),
    formatter=DeepSeekChatFormatter(),  # 格式化器
    toolkit=toolkit,
    memory=InMemoryMemory(),
)


async def example_router_implicit() -> None:
    """Example of implicit routing with tool calls."""
    msg_user = Msg("user", "Help me to generate a quick sort function in Python", "user", )

    # Route the query
    await router_implicit(msg_user)


asyncio.run(example_router_implicit())
