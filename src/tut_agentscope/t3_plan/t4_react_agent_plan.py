import asyncio

from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.plan import PlanNotebook, SubTask

model_name = "deepseek-chat"
api_key = "sk-bf45e97095c64d8aa0336a7857563493"
base_url = "https://api.deepseek.com"


async def main() -> None:
    plan_notebook = PlanNotebook()

    """手动创建计划：计算水分子 (H2O) 的摩尔质量"""
    await plan_notebook.create_plan(
        name="Molecular Molar Mass Calculation",
        description="Conduct a step-by-step calculation of the molar mass for water (H2O) using basic atomic masses.",
        expected_outcome="A final report with the molar mass of H2O in g/mol.",
        subtasks=[
            SubTask(
                name="Recall atomic masses",
                description="List the atomic masses: Hydrogen (H) = 1 g/mol, Oxygen (O) = 16 g/mol.",
                expected_outcome="Atomic masses: H=1, O=16",
            ),
            SubTask(
                name="Count atoms in molecule",
                description="For H2O: 2 Hydrogen atoms and 1 Oxygen atom.",
                expected_outcome="Atom counts: H=2, O=1",
            ),
            SubTask(
                name="Calculate total mass",
                description="Compute: (2 * 1) + (1 * 16) = 18 g/mol.",
                expected_outcome="Molar mass = 18 g/mol",
            ),
            SubTask(
                name="Write report",
                description="Summarize the calculation in a simple text report.",
                expected_outcome="Report: The molar mass of H2O is 18 g/mol.",
            ),
        ],
    )

    print("The current hint message:\n")
    msg = await plan_notebook.get_current_hint()
    print(f"{msg.name}: {msg.content}")

    agent = ReActAgent(
        name="Friday",
        sys_prompt="You are a helpful assistant. Use basic math only, no external tools needed.",
        model=OpenAIChatModel(api_key=api_key, model_name=model_name, client_args={"base_url": base_url}, stream=False),
        formatter=DeepSeekChatFormatter(),
        plan_notebook=plan_notebook,
        memory=InMemoryMemory(),
    )

    user = UserAgent(name="user")

    msg = None
    while True:
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break
        msg = await agent(msg)
        agent_response = msg.get_text_content()
        print(f"Agent response: {agent_response}")


if __name__ == '__main__':
    asyncio.run(main())
