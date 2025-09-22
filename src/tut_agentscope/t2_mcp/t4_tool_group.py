import asyncio

import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import (
    DeepSeekChatFormatter,
)
from agentscope.mcp import HttpStatefulClient, HttpStatelessClient
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

model_name = "deepseek-chat"
api_key = "sk-bf45e97095c64d8aa0336a7857563493"
base_url = "https://api.deepseek.com"


async def main():
    agentscope.init(
        project='tut-agentscope',
        name='t1_react_agent',
        studio_url="http://localhost:8000"
    )

    toolkit = Toolkit()
    toolkit.create_tool_group(group_name="t1_tool_group", description="t1_tool_group", active=True, notes="这是t1_tool_group，用于测试")

    client1 = HttpStatefulClient(name="t1_react_agent_1", transport='sse', url="http://localhost:8000")
    await client1.connect()
    # 注册MCP客户端到工具组
    await toolkit.register_mcp_client(client1, group_name="t1_tool_group")

    client2 = HttpStatelessClient(name="t1_react_agent_2", transport='sse', url="http://localhost:8000")
    # 注册MCP客户端到工具组
    await toolkit.register_mcp_client(client2, group_name="t1_tool_group")

    print("MCP clients: ", toolkit.get_json_schemas())

    agent = ReActAgent(
        name="Friday",
        sys_prompt="You're a helpful assistant named Friday",
        model=OpenAIChatModel(model_name=model_name, api_key=api_key, client_args={"base_url": base_url}),
        formatter=DeepSeekChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=Toolkit(),
    )

    """Interact with the plan agent."""
    user = UserAgent(name="user")

    msg = None
    while True:
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break
        msg = await agent(msg)


if __name__ == '__main__':
    asyncio.run(main())
