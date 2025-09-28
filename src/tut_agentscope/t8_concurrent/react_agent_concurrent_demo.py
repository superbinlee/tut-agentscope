import asyncio

import agentscope
from agentscope.agent import ReActAgent
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

model_name = "deepseek-chat"
api_key = "sk-bf45e97095c64d8aa0336a7857563493"
base_url = "https://api.deepseek.com"

agentscope.init(project='concurrent_react_demo', name='t1_react_agent', studio_url="http://localhost:8000")


async def run_concurrent_agents() -> None:
    """并发运行两个 ReActAgent。"""
    agent1 = ReActAgent(
        name="Friday",
        sys_prompt="You're a helpful assistant named Friday",
        model=OpenAIChatModel(model_name=model_name, api_key=api_key, client_args={"base_url": base_url}),
        formatter=DeepSeekChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=Toolkit(),
    )
    agent2 = ReActAgent(
        name="Friday",
        sys_prompt="You're a helpful assistant named Friday",
        model=OpenAIChatModel(model_name=model_name, api_key=api_key, client_args={"base_url": base_url}),
        formatter=DeepSeekChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=Toolkit(),
    )
    # 构造用户消息
    msg1 = Msg(name="user", content="What is 123 multiplied by 456?", role="user")
    msg2 = Msg(name="user", content="What is the capital of Canada?", role="user")

    # 并发执行两个智能体
    responses = await asyncio.gather(
        agent1(msg1),
        agent2(msg2),
    )

    # 打印结果
    print("\n" + "=" * 50)
    print("Final Responses:")
    print("=" * 50)
    for i, resp in enumerate(responses, 1):
        print(f"\n--- Agent {i} ---")
        print(resp.content)


if __name__ == "__main__":
    asyncio.run(run_concurrent_agents())
