import asyncio

from agentscope.agent import ReActAgent
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import ToolResponse, Toolkit, execute_python_code

model_name = "deepseek-chat"
api_key = "sk-bf45e97095c64d8aa0336a7857563493"
base_url = "https://api.deepseek.com"


async def create_worker(task_description: str) -> ToolResponse:
    """Create a worker to finish the given task. The worker is equipped with python execution tool.

    Args:
        task_description (``str``):
            The description of the task to be finished by the worker.
    """
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)

    worker = ReActAgent(
        name="Worker",
        sys_prompt="You're a worker agent. Your target is to finish the given task.",
        model=OpenAIChatModel(model_name=model_name, api_key=api_key, client_args={"base_url": base_url}),
        formatter=DeepSeekChatFormatter(),
        toolkit=toolkit,
    )
    res = await worker(Msg("user", task_description, "user"))
    return ToolResponse(content=res.get_content_blocks("text"), stream=True)


async def run_handoffs() -> None:
    """Example of handoffs workflow."""
    toolkit = Toolkit()
    toolkit.register_tool_function(create_worker)

    orchestrator = ReActAgent(
        name="Orchestrator",
        sys_prompt="You're an orchestrator agent. Your target is to finish the given task by decomposing it into smaller tasks and creating workers to finish them.",
        model=OpenAIChatModel(model_name=model_name, api_key=api_key, client_args={"base_url": base_url}),
        memory=InMemoryMemory(),
        formatter=DeepSeekChatFormatter(),
        toolkit=toolkit,
    )

    task_description = "Execute hello world in Python"
    await orchestrator(Msg("user", task_description, "user"))


asyncio.run(run_handoffs())
