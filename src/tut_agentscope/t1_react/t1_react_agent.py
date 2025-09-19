import asyncio
import agentscope
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import (
    DeepSeekChatFormatter,
)
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

model_name = "deepseek-chat"
api_key = "sk-bf45e97095c64d8aa0336a7857563493"
base_url = "https://api.deepseek.com"

agentscope.init(
    project='tut-agentscope',
    name='t1_react_agent',
    studio_url="http://localhost:3000"
)

agent = ReActAgent(
    name="Friday",
    sys_prompt="You're a helpful assistant named Friday",
    model=OpenAIChatModel(model_name=model_name, api_key=api_key, client_args={"base_url": base_url}),
    formatter=DeepSeekChatFormatter(),
    memory=InMemoryMemory(),
    toolkit=Toolkit(),
)


async def interact_with_agent() -> None:
    """Interact with the plan agent."""
    user = UserAgent(name="user")

    msg = None
    while True:
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break
        msg = await agent(msg)


asyncio.run(interact_with_agent())
