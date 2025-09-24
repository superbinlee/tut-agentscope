import asyncio
import json
from typing import Literal

# 导入Agentscope相关组件
from agentscope.agent import ReActAgent
from agentscope.formatter import DeepSeekChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from pydantic import BaseModel, Field

# 模型配置参数
model_name = "deepseek-chat"
api_key = "sk-bf45e97095c64d8aa0336a7857563493"
base_url = "https://api.deepseek.com"


class RoutingChoice(BaseModel):
    """
    路由选择模型，用于定义智能体可以选择的后续任务类型。
    基于Pydantic模型，提供结构化输出验证。
    """
    your_choice: Literal["Content Generation", "Programming", "Information Retrieval", None,] = Field(
        description="选择适当的后续任务类型，如果任务太简单或没有合适的任务类型则选择``None``",
    )
    task_description: str | None = Field(
        description="任务描述",
        default=None,
    )


async def main() -> None:
    """
    主函数，演示如何使用ReActAgent进行任务路由。
    该函数创建一个路由智能体，并使用结构化模型获取路由结果。
    """
    # 创建路由智能体
    router_agent = ReActAgent(
        name="Router",
        sys_prompt="You're a routing agent. Your target is to route the user query to the right follow-up task.",
        model=OpenAIChatModel(
            api_key=api_key,
            model_name=model_name,
            client_args={"base_url": base_url},
            stream=False
        ),
        formatter=DeepSeekChatFormatter(),
        memory=InMemoryMemory(),
    )

    # 创建用户消息，要求写一首古代风格的诗
    msg_user = Msg("user", "Help me to write a poem, 要求这首诗是古代风格。", "user")

    # 使用结构化模型进行查询路由
    msg_res = await router_agent(msg_user, structured_model=RoutingChoice)

    # 结构化输出存储在metadata字段中，打印结果
    print("结构化输出结果:")
    print(json.dumps(msg_res.metadata, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
