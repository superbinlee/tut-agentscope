# -*- coding: utf-8 -*-
"""Deep Research Agent"""
# pylint: disable=too-many-lines, no-name-in-module
import os
import json
import asyncio

from typing import Type, Optional, Any, Tuple
from datetime import datetime
from copy import deepcopy
import shortuuid
from pydantic import BaseModel

from built_in_prompt.promptmodule import (
    SubtasksDecomposition,
    WebExtraction,
    FollowupJudge,
    ReflectFailure,
)
from utils import (
    truncate_search_result,
    load_prompt_dict,
    get_dynamic_tool_call_json,
    get_structure_output,
)

from agentscope import logger, setup_logger
from agentscope.mcp import StatefulClientBase
from agentscope.agent import ReActAgent
from agentscope.model import ChatModelBase
from agentscope.formatter import FormatterBase
from agentscope.memory import MemoryBase
from agentscope.tool import (
    ToolResponse,
    view_text_file,
    write_text_file,
)
from agentscope.message import (
    Msg,
    ToolUseBlock,
    TextBlock,
    ToolResultBlock,
)

_DEEP_RESEARCH_AGENT_DEFAULT_SYS_PROMPT = "You're a helpful assistant."

_LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
_LOG_PATH = os.path.join(
    _LOG_DIR,
    f"log_{datetime.now().strftime('%y%m%d%H%M%S')}.md",
)
os.makedirs(_LOG_DIR, exist_ok=True)
setup_logger(level="INFO", filepath=_LOG_PATH)


class SubTaskItem(BaseModel):
    """Subtask item of deep research agent."""

    objective: str
    working_plan: Optional[str] = None
    knowledge_gaps: Optional[str] = None


class DeepResearchAgent(ReActAgent):
    """
    Deep Research Agent for sophisticated research tasks.

    Example:
        .. code-block:: python

        agent = DeepResearchAgent(
            name="Friday",
            sys_prompt="You are a helpful assistant named Friday.",
            model=my_chat_model,
            formatter=my_chat_formatter,
            memory=InMemoryMemory(),
            search_mcp_client=my_tavily_search_client,
            tmp_file_storage_dir=agent_working_dir,
        )
        response = await agent(
            Msg(
                name=“user”,
                content="Please give me a survey of the LLM-empowered agent.",
                role=“user”
            )
        )
        ```
    """

    def __init__(
            self,
            name: str,
            model: ChatModelBase,
            formatter: FormatterBase,
            memory: MemoryBase,
            search_mcp_client: StatefulClientBase,
            sys_prompt: str = _DEEP_RESEARCH_AGENT_DEFAULT_SYS_PROMPT,
            max_iters: int = 30,
            max_depth: int = 3,
            tmp_file_storage_dir: str = "tmp",
    ) -> None:
        """Initialize the Deep Research Agent.

        Args:
            name (str):
                The unique identifier name for the agent instance.
            model (ChatModelBase):
                The chat model used for generating responses and reasoning.
            formatter (FormatterBase):
                The formatter used to convert messages into the required
                format for the model API.
            memory (MemoryBase):
                The memory component used to store and retrieve dialogue
                history.
            search_mcp_client (StatefulClientBase):
                The mcp client used to provide the tools for deep search.
            sys_prompt (str, optional):
                The system prompt that defines the agent's behavior
                and personality.
                Defaults to _DEEP_RESEARCH_AGENT_DEFAULT_SYS_PROMPT.
            max_iters (int, optional):
                The maximum number of reasoning-acting loop iterations.
                Defaults to 30.
            max_depth (int, optional):
                The maximum depth of query expansion during deep searching.
                Defaults to 3.
            tmp_file_storage_dir (str, optional):
                The storage dir for generated files.
                Default to 'tmp'
        Returns:
            None
        """

        # initialization of prompts
        self.prompt_dict = load_prompt_dict()

        # Enhance the system prompt for deep research agent
        add_note = self.prompt_dict["add_note"].format_map(
            {"finish_function_name": f"`{self.finish_function_name}`"},
        )
        tool_use_rule = self.prompt_dict["tool_use_rule"].format_map(
            {"tmp_file_storage_dir": tmp_file_storage_dir},
        )
        sys_prompt = f"{sys_prompt}\n{add_note}\n{tool_use_rule}"

        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=formatter,
            memory=memory,
            max_iters=max_iters,
        )
        # 设置最大递归深度，用于控制子任务分解的层级
        self.max_depth = max_depth
        # 保存传入的记忆组件引用
        self.memory = memory
        # 设置临时文件存储目录，用于保存中间报告和最终报告
        self.tmp_file_storage_dir = tmp_file_storage_dir
        # 初始化当前子任务列表，用于跟踪任务分解的层次结构
        self.current_subtask = []

        # 注册所有必要的工具函数到工具包中
        # 注册查看文本文件的工具函数
        self.toolkit.register_tool_function(view_text_file)
        # 注册写入文本文件的工具函数
        self.toolkit.register_tool_function(write_text_file)
        # 异步注册MCP客户端，用于提供搜索功能
        asyncio.get_running_loop().create_task(
            self.toolkit.register_mcp_client(search_mcp_client),
        )

        # 定义工具函数名称常量，便于后续调用
        # 搜索功能函数名
        self.search_function = "tavily-search"
        # 网页内容提取功能函数名
        self.extract_function = "tavily-extract"
        # 读取文件功能函数名
        self.read_file_function = "view_text_file"
        # 写入文件功能函数名
        self.write_file_function = "write_text_file"
        # 总结中间结果功能函数名
        self.summarize_function = "summarize_intermediate_results"

        # 初始化中间记忆存储，用于保存推理和执行过程中的临时信息
        self.intermediate_memory = []
        # 基于代理名称和当前时间生成报告文件名前缀，确保唯一性
        self.report_path_based = self.name + datetime.now().strftime(
            "%y%m%d%H%M%S",
        )
        # 初始化报告索引，用于编号生成的中间报告
        self.report_index = 1
        # 存储所需的结构化模型类型（如果有的话）
        self._required_structured_model = None
        # 存储用户的原始查询内容
        self.user_query = None

        # 将自定义函数添加到工具包中
        # 注册失败反思函数，用于在执行失败时进行反思和调整
        self.toolkit.register_tool_function(self.reflect_failure)
        # 注册中间结果总结函数，用于总结已完成的步骤
        self.toolkit.register_tool_function(
            self.summarize_intermediate_results,
        )

    async def reply(
            self,
            msg: Msg | list[Msg] | None = None,
            structured_model: Type[BaseModel] | None = None,
    ) -> Msg:
        """代理的回复方法"""

        # 维护子任务列表：提取用户查询内容并作为初始子任务添加
        self.user_query = msg.get_text_content()
        self.current_subtask.append(
            SubTaskItem(objective=self.user_query),
        )
        # 1. 识别预期输出并生成计划：分解初始任务，获取知识缺口
        # 2. 该函数会分析当前任务，确定完成任务所需的知识缺口，并制定详细的工作计划
        # 3. 工作计划将包含具体的步骤和策略，用于指导后续的搜索和信息收集
        await self.decompose_and_expand_subtask()
        # 将知识缺口（预期输出）附加到原始消息内容中
        msg.content += f"\nExpected Output:\n{self.current_subtask[0].knowledge_gaps}"

        # 将用户查询消息添加到记忆中
        await self.memory.add(msg)

        # 如果提供了结构化模型，则记录该模型并设置扩展模型
        if structured_model:
            self._required_structured_model = structured_model
            self.toolkit.set_extended_model(self.finish_function_name, structured_model)

        # 主循环：最多执行max_iters次迭代
        for _ in range(self.max_iters):
            # 首先生成工作计划：如果当前子任务没有工作计划，则生成一个
            if not self.current_subtask[-1].working_plan:
                await self.decompose_and_expand_subtask()

            # 构建推理指令：准备推理所需的提示信息
            cur_plan = self.current_subtask[-1].working_plan
            cur_know_gap = self.current_subtask[-1].knowledge_gaps
            reasoning_prompt = self.prompt_dict["reasoning_prompt"].format_map(
                {
                    "objective": self.current_subtask[-1].objective,  # 当前目标
                    "plan": cur_plan
                    if cur_plan
                    else "There is no working plan now.",  # 当前计划
                    "knowledge_gap": f"## Knowledge Gaps:\n {cur_know_gap}"  # 知识缺口
                    if cur_know_gap
                    else "",
                    "depth": len(self.current_subtask),  # 当前任务深度
                },
            )
            # 创建推理提示消息并添加到中间记忆
            reasoning_prompt_msg = Msg(
                "user",
                content=[
                    TextBlock(
                        type="text",
                        text=reasoning_prompt,
                    ),
                ],
                role="user",
            )
            self.intermediate_memory.append(reasoning_prompt_msg)

            # 推理生成工具调用：备份记忆，添加中间记忆，执行推理
            backup_memory = deepcopy(self.memory)  # 备份当前记忆
            await self.memory.add(self.intermediate_memory)  # 添加中间记忆
            msg_reasoning = await self._reasoning()  # 执行推理
            self.memory = backup_memory  # 恢复记忆状态

            # 调用工具：遍历所有生成的工具调用并执行
            for tool_call in msg_reasoning.get_content_blocks("tool_use"):
                # 将工具调用添加到中间记忆
                self.intermediate_memory.append(
                    Msg(
                        self.name,
                        content=[tool_call],
                        role="assistant",
                    ),
                )
                # 执行工具调用
                msg_response = await self._acting(tool_call)
                # 如果工具调用返回响应消息，则添加到记忆并返回
                if msg_response:
                    await self.memory.add(msg_response)
                    self.current_subtask = []  # 清空子任务列表
                    return msg_response

        # 当达到最大迭代次数时，总结所有发现并返回
        return await self._summarizing()

    async def _acting(self, tool_call: ToolUseBlock) -> Msg | None:
        """
        执行工具调用并处理其响应，特别针对浏览器相关操作进行处理。

        该方法负责执行传入的工具调用，处理工具返回的结果，并根据不同的工具类型进行相应的后处理。
        它会处理工具调用的整个生命周期，包括执行、结果处理、内存更新等操作。

        Args:
            tool_call (ToolUseBlock):
                工具使用块，包含工具名称、参数和用于执行的唯一标识符。

        Returns:
            Msg | None:
                如果成功调用了完成函数（finish function），则返回响应消息；
                否则返回None以继续推理-执行循环。
        """

        # 创建工具结果消息模板，用于存储工具执行结果
        tool_res_msg = Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],  # 工具调用ID
                    name=tool_call["name"],  # 工具名称
                    output=[],  # 初始化为空输出
                ),
            ],
            "system",
        )

        # 初始化变量，用于跟踪是否需要更新内存和存储中间报告
        update_memory = False
        intermediate_report = ""
        chunk = ""

        try:
            # 执行工具调用，获取工具响应
            tool_res = await self.toolkit.call_tool_function(tool_call)

            # 异步处理工具响应的生成器（流式处理）
            async for chunk in tool_res:
                # 将当前块的内容更新到工具结果消息中
                tool_res_msg.content[0][  # type: ignore[index]
                    "output"
                ] = chunk.content

                # 控制工具调用结果的打印行为
                # 如果不是完成函数调用，或者虽然是完成函数调用但执行失败，则打印结果
                if (
                        tool_call["name"] != self.finish_function_name
                        or tool_call["name"] == self.finish_function_name
                        and not chunk.metadata.get("success")
                ):
                    await self.print(tool_res_msg, chunk.is_last)

                # 检查是否成功调用了完成函数（generate_response）
                if tool_call[
                    "name"
                ] == self.finish_function_name and chunk.metadata.get(
                    "success",
                    True,
                ):
                    # 如果当前没有子任务，说明是最终完成，返回响应消息
                    if len(self.current_subtask) == 0:
                        return chunk.metadata.get("response_msg")

                # 处理中间结果总结工具的调用
                elif tool_call["name"] == self.summarize_function:
                    # 清空中间记忆
                    self.intermediate_memory = []
                    # 将总结结果添加到主记忆中
                    await self.memory.add(
                        Msg(
                            "assistant",
                            [
                                TextBlock(
                                    type="text",
                                    text=chunk.content[0]["text"],  # 提取总结文本
                                ),
                            ],
                            "assistant",
                        ),
                    )

                # 处理搜索和提取工具的结果，截断过长的搜索结果
                elif tool_call["name"] in [
                    self.search_function,
                    self.extract_function,
                ]:
                    tool_res_msg.content[0]["output"] = truncate_search_result(
                        tool_res_msg.content[0]["output"],
                    )

                # 检查是否需要更新内存（当有中间报告生成时）
                if isinstance(chunk.metadata, dict) and chunk.metadata.get(
                        "update_memory",
                ):
                    update_memory = True
                    intermediate_report = chunk.metadata.get(
                        "intermediate_report",
                    )
            # 工具执行完成，返回None继续循环
            return None

        # 无论是否出现异常，都会执行的清理代码
        finally:
            # 将工具结果消息记录到中间记忆中（除了总结函数）
            if tool_call["name"] != self.summarize_function:
                self.intermediate_memory.append(tool_res_msg)

            # 如果是搜索工具调用，执行后续处理（可能需要从网页提取更多信息）
            if tool_call["name"] == self.search_function:
                # 调用_follow_up方法进行后续处理
                extract_res = await self._follow_up(chunk.content, tool_call)
                # 如果需要更新内存，则清空中间记忆并添加新的内容
                if isinstance(
                        extract_res.metadata,
                        dict,
                ) and extract_res.metadata.get("update_memory"):
                    self.intermediate_memory = []
                    await self.memory.add(
                        Msg(
                            "assistant",
                            content=[
                                TextBlock(
                                    type="text",
                                    text=extract_res.metadata.get(
                                        "intermediate_report",
                                    ).content[0]["text"],
                                ),
                            ],
                            role="assistant",
                        ),
                    )

            # 如果需要更新内存（有中间报告生成），则清空中间记忆并添加报告内容
            if update_memory:
                self.intermediate_memory = []
                await self.memory.add(
                    Msg(
                        "assistant",
                        content=[
                            TextBlock(
                                type="text",
                                text=intermediate_report.content[0]["text"],
                            ),
                        ],
                        role="assistant",
                    ),
                )

    async def get_model_output(
            self,
            msgs: list,
            format_template: Type[BaseModel] = None,
            stream: bool = True,
    ) -> Any:
        """
        调用大语言模型并获取结构化或非结构化输出。

        这个方法是调用大模型的核心接口，根据是否提供格式模板来决定是否进行结构化输出：
        1. 如果提供了 format_template，则调用模型的结构化输出功能，强制模型按照指定的数据结构返回结果
        2. 如果没有提供 format_template，则进行普通的文本对话

        Args:
            msgs (list): 包含对话历史的消息列表，每个消息通常包含角色(role)和内容(content)
            format_template (BaseModel): 可选的Pydantic模型类，用于指定期望的结构化输出格式
            stream (bool): 是否使用流式输出，默认为True

        调用大模型的关键代码行：
        - `res = await self.model(...)` 这里实际调用了大语言模型
        - `await self.formatter.format(msgs=msgs)` 格式化消息以适配模型输入要求
        - `tools=get_dynamic_tool_call_json(format_template)` 当需要结构化输出时生成工具调用规范
        - `get_structure_output(blocks)` 解析并验证模型返回的结构化数据
        """
        blocks = None

        # 判断是否需要结构化输出
        if format_template:
            # 调用大语言模型，传入格式化后的消息和工具调用规范以获得结构化输出
            res = await self.model(
                args=await self.formatter.format(msgs=msgs),
                tools=get_dynamic_tool_call_json(format_template, ),
            )

            # 处理流式或非流式响应
            if stream:
                # 流式输出：逐块获取内容
                async for content_chunk in res:
                    blocks = content_chunk.content
            else:
                # 非流式输出：一次性获取全部内容
                blocks = res.content

            # 解析并返回结构化输出
            return get_structure_output(blocks)
        else:
            # 普通对话模式：直接调用大语言模型
            res = await self.model(args=await self.formatter.format(msgs=msgs), )

            # 处理流式或非流式响应
            if stream:
                # 流式输出：逐块获取内容
                async for content_chunk in res:
                    blocks = content_chunk.content
            else:
                # 非流式输出：一次性获取全部内容
                blocks = res.content

            # 返回原始内容
            return blocks

    async def call_specific_tool(
            self,
            func_name: str,
            params: dict = None,
    ) -> Tuple[Msg, Msg]:
        """
        Call the specific tool in toolkit.

        Args:
            func_name (str): name of the tool.
            params (dict): input parameters of the tool.
        """
        tool_call = ToolUseBlock(
            id=shortuuid.uuid(),
            type="tool_use",
            name=func_name,
            input=params,
        )
        tool_call_msg = Msg(
            "assistant",
            [tool_call],
            role="assistant",
        )

        # get tool acting res
        tool_res_msg = Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=[],
                ),
            ],
            "system",
        )
        tool_res = await self.toolkit.call_tool_function(
            tool_call,
        )
        async for chunk in tool_res:
            tool_res_msg.content[0]["output"] = chunk.content

        return tool_call_msg, tool_res_msg

    async def decompose_and_expand_subtask(self) -> ToolResponse:
        """识别当前子任务的知识缺口并生成工作计划。

        通过子任务分解来生成工作计划，工作计划包括完成任务的必要步骤和扩展步骤。

        Returns:
            ToolResponse: 当前子任务的知识缺口和工作计划（JSON格式）
        """
        # 检查当前子任务深度是否超过最大深度限制
        if len(self.current_subtask) <= self.max_depth:
            # 获取任务分解的系统提示词
            decompose_sys_prompt = self.prompt_dict["decompose_sys_prompt"]

            # 收集历史子任务的工作计划，用于上下文参考
            # 这样做是为了让模型了解之前的决策和计划，从而更好地制定当前的子任务计划
            # 保证任务分解的一致性和连贯性
            previous_plan = ""
            for i, subtask in enumerate(self.current_subtask):
                previous_plan += f"The {i}-th plan: {subtask.working_plan}\n"

            # 构建包含历史计划和当前目标的用户提示词
            # 将历史计划和当前目标传递给模型，以便它能够基于之前的上下文进行决策
            previous_plan_inst = self.prompt_dict["previous_plan_inst"].format_map(
                {
                    "previous_plan": previous_plan,  # 历史计划
                    "objective": self.current_subtask[-1].objective,  # 当前目标
                },
            )

            try:
                # 调用模型获取知识缺口和工作计划（使用结构化输出格式）
                gaps_and_plan = await self.get_model_output(
                    msgs=[
                        Msg("system", decompose_sys_prompt, "system"),  # 系统提示
                        Msg("user", previous_plan_inst, "user"),  # 用户提示
                    ],
                    format_template=SubtasksDecomposition,  # 强制结构化输出格式
                    stream=self.model.stream,  # 流式输出
                )
                # 将结果转换为格式化的JSON字符串
                response = json.dumps(
                    gaps_and_plan,
                    indent=2,
                    ensure_ascii=False,
                )
            except Exception:  # noqa: F841
                # 如果模型调用失败，设置空结果和重试提示
                gaps_and_plan = {}
                response = self.prompt_dict["retry_hint"].format_map(
                    {"state": "decomposing the subtask"},  # 重试提示状态
                )

            # 更新当前子任务的知识缺口和工作计划
            self.current_subtask[-1].knowledge_gaps = gaps_and_plan.get(
                "knowledge_gaps",
                None,
            )
            self.current_subtask[-1].working_plan = gaps_and_plan.get(
                "working_plan",
                None,
            )

            # 返回包含结果的工具响应
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=response,
                    ),
                ],
            )

        # 如果超过最大深度限制，返回深度限制提示
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=self.prompt_dict["max_depth_hint"],
                ),
            ],
        )

    async def _follow_up(
            self,
            search_results: list | str,
            tool_call: ToolUseBlock,
    ) -> ToolResponse:
        """
        深入阅读网站以挖掘更多与任务相关的信息，并在必要时生成后续子任务进行深度搜索。

        该方法是深度研究过程中的关键组件，负责分析现有搜索结果，判断是否需要进一步获取信息，
        如果需要，则调用相关工具进行深度信息提取。

        Args:
            search_results (list | str): 之前搜索获取的结果，可以是字符串或列表形式
            tool_call (ToolUseBlock): 工具调用信息，包含查询参数等

        Returns:
            ToolResponse: 包含处理结果的工具响应对象
        """

        # 检查当前子任务深度是否小于最大深度限制
        if len(self.current_subtask) < self.max_depth:
            # 第一步：查询扩展 - 基于现有结果和目标扩展搜索需求
            expansion_sys_prompt = self.prompt_dict["expansion_sys_prompt"]
            expansion_inst = self.prompt_dict["expansion_inst"].format_map(
                {
                    "objective": tool_call["input"].get("query", ""),  # 从工具调用中获取查询目标
                    "checklist": self.current_subtask[0].knowledge_gaps,  # 初始任务的知识缺口
                    "knowledge_gaps": self.current_subtask[-1].working_plan,  # 当前子任务的工作计划
                    "search_results": search_results,  # 已有搜索结果
                },
            )

            try:
                # 调用模型获取扩展子任务信息（使用WebExtraction结构化输出格式）
                follow_up_subtask = await self.get_model_output(
                    msgs=[
                        Msg("system", expansion_sys_prompt, "system"),  # 系统提示词
                        Msg("user", expansion_inst, "user"),  # 包含所有上下文信息的用户提示词
                    ],
                    format_template=WebExtraction,  # 强制使用结构化输出格式
                    stream=self.model.stream,  # 保持流式输出设置
                )
            except Exception:  # noqa: F841
                # 发生异常时，使用空的后续子任务
                follow_up_subtask = {}

            # 第二步：提取URL - 从模型响应中获取需要进一步访问的URL
            if follow_up_subtask.get("need_more_information", False):
                # 构建包含推理过程的响应消息
                expansion_response_msg = Msg(
                    "assistant",
                    follow_up_subtask.get(
                        "reasoning",
                        "I need more information.",  # 默认推理说明
                    ),
                    role="assistant",
                )
                urls = follow_up_subtask.get("url", None)  # 从响应中提取URL
                logger.info("Reading %s", urls)  # 记录正在读取的URL

                # 调用信息提取工具
                params = {
                    "urls": urls,  # 要提取信息的URL
                    "extract_depth": "basic",  # 提取深度设为基础
                }
                # 调用特定的信息提取工具函数
                (
                    extract_tool_use_msg,
                    extract_tool_res_msg,
                ) = await self.call_specific_tool(
                    func_name=self.extract_function,  # 使用配置的提取函数
                    params=params,  # 提取参数
                )
                # 将工具使用消息添加到中间记忆中
                self.intermediate_memory.append(extract_tool_use_msg)

                # 对提取结果进行截断处理，避免结果过大
                extract_tool_res_msg.content[0][
                    "output"
                ] = truncate_search_result(
                    extract_tool_res_msg.content[0]["output"],
                )
                # await self.memory.add(tool_res_msg)  # 注释掉的代码：将结果添加到记忆中

    async def summarize_intermediate_results(self) -> ToolResponse:
        """Summarize the intermediate results into a report when a step
        in working plan is completed.

        Returns:
            ToolResponse:
                The summarized draft report.
        """
        if len(self.intermediate_memory) == 0:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=self.prompt_dict["no_result_hint"],
                    ),
                ],
            )
        # agent actively call this tool
        if self.intermediate_memory[-1].name == self.summarize_function:
            blocks = await self.get_model_output(
                msgs=self.intermediate_memory
                     + [
                         Msg(
                             "user",
                             self.prompt_dict["summarize_hint"].format_map(
                                 {
                                     "plan": self.current_subtask[-1].working_plan,
                                 },
                             ),
                             role="user",
                         ),
                     ],
                stream=self.model.stream,
            )
            self.current_subtask[-1].working_plan = blocks[0][
                "text"
            ]  # type: ignore[index]
        report_prefix = "#" * len(self.current_subtask)
        summarize_sys_prompt = self.prompt_dict[
            "summarize_sys_prompt"
        ].format_map(
            {"report_prefix": report_prefix},
        )
        # get all tool result
        tool_result = ""
        for item in self.intermediate_memory:
            if isinstance(item.content, str):
                tool_result += item.content + "\n"
            elif isinstance(item.content, list):
                for each in item.content:
                    if each["type"] == "tool_result":
                        tool_result += str(each) + "\n"
            else:
                logger.warning(
                    "Unknown content type: %s!",
                    type(item.content),
                )
                continue
        summarize_instruction = self.prompt_dict["summarize_inst"].format_map(
            {
                "objective": self.current_subtask[0].objective,
                "knowledge_gaps": self.current_subtask[0].knowledge_gaps,
                "working_plan": self.current_subtask[-1].working_plan,
                "tool_result": tool_result,
            },
        )

        blocks = await self.get_model_output(
            msgs=[
                Msg("system", summarize_sys_prompt, "system"),
                Msg("user", summarize_instruction, "user"),
            ],
            stream=self.model.stream,
        )
        intermediate_report = blocks[0]["text"]  # type: ignore[index]

        # Write the intermediate report
        intermediate_report_path = os.path.join(
            self.tmp_file_storage_dir,
            f"{self.report_path_based}_"
            f"inprocess_report_{self.report_index}.md",
        )
        self.report_index += 1
        params = {
            "file_path": intermediate_report_path,
            "content": intermediate_report,
        }
        await self.call_specific_tool(
            func_name=self.write_file_function,
            params=params,
        )
        logger.info(
            "Storing the intermediate findings: %s",
            intermediate_report,
        )
        if (
                self.intermediate_memory[-1].has_content_blocks("tool_use")
                and self.intermediate_memory[-1].get_content_blocks("tool_use")[0][
            "name"
        ]
                == self.summarize_function
        ):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=self.prompt_dict["update_report_hint"].format_map(
                            {
                                "intermediate_report": intermediate_report,
                                "report_path": intermediate_report_path,
                            },
                        ),
                    ),
                ],
            )
        else:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=self.prompt_dict["save_report_hint"].format_map(
                            {
                                "intermediate_report": intermediate_report,
                            },
                        ),
                    ),
                ],
            )

    async def _generate_deepresearch_report(
            self,
            checklist: str,
    ) -> Tuple[Msg, str]:
        """Collect and polish all draft reports into a final report.

        Args:
            checklist (`str`):
                The expected output items of the original task.
        """
        reporting_sys_prompt = self.prompt_dict["reporting_sys_prompt"]
        reporting_sys_prompt.format_map(
            {
                "original_task": self.user_query,
                "checklist": checklist,
            },
        )

        # Collect all intermediate reports
        if self.report_index > 1:
            inprocess_report = ""
            for index in range(self.report_index):
                params = {
                    "file_path": os.path.join(
                        self.tmp_file_storage_dir,
                        f"{self.report_path_based}_"
                        f"inprocess_report_{index + 1}.md",
                    ),
                }
                _, read_draft_tool_res_msg = await self.call_specific_tool(
                    func_name=self.read_file_function,
                    params=params,
                )
                inprocess_report += (
                        read_draft_tool_res_msg.content[0]["output"][0]["text"]
                        + "\n"
                )

            msgs = [
                Msg(
                    "system",
                    content=reporting_sys_prompt,
                    role="system",
                ),
                Msg(
                    "user",
                    content=f"Draft report:\n{inprocess_report}",
                    role="user",
                ),
            ]
        else:  # Use only intermediate memory to generate report
            msgs = [
                       Msg(
                           "system",
                           content=reporting_sys_prompt,
                           role="system",
                       ),
                   ] + self.intermediate_memory

        blocks = await self.get_model_output(
            msgs=msgs,
            stream=self.model.stream,
        )
        final_report_content = blocks[0]["text"]  # type: ignore[index]
        logger.info(
            "The final Report is generated: %s",
            final_report_content,
        )

        # Write the final report into a file
        detailed_report_path = os.path.join(
            self.tmp_file_storage_dir,
            f"{self.report_path_based}_detailed_report.md",
        )

        params = {
            "file_path": detailed_report_path,
            "content": final_report_content,
        }
        _, write_report_tool_res_msg = await self.call_specific_tool(
            func_name=self.write_file_function,
            params=params,
        )

        return write_report_tool_res_msg, detailed_report_path

    async def _summarizing(self) -> Msg:
        """Generate a report based on the exsisting findings when the
        agent fails to solve the problem in the maximum iterations."""

        (
            summarized_content,
            _,
        ) = await self._generate_deepresearch_report(
            checklist=self.current_subtask[0].knowledge_gaps,
        )
        return Msg(
            name=self.name,
            role="assistant",
            content=json.dumps(
                summarized_content.content[0]["output"][0],
                indent=2,
                ensure_ascii=False,
            ),
        )

    async def reflect_failure(self) -> ToolResponse:
        """Reflect on the failure of the action and determine to rephrase
        the plan or deeper decompose the current step.

        Returns:
            ToolResponse:
                The reflection about plan rephrasing and subtask decomposition.
        """
        reflect_sys_prompt = self.prompt_dict["reflect_sys_prompt"]
        conversation_history = ""
        for msg in self.intermediate_memory:
            conversation_history += (
                    json.dumps(
                        {"role": "user", "content": msg.content},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
            )
        reflect_inst = self.prompt_dict["reflect_instruction"].format_map(
            {
                "conversation_history": conversation_history,
                "plan": self.current_subtask[-1].working_plan,
            },
        )
        try:
            reflection = await self.get_model_output(
                msgs=[
                    Msg("system", reflect_sys_prompt, "system"),
                    Msg("user", reflect_inst, "user"),
                ],
                format_template=ReflectFailure,
                stream=self.model.stream,
            )
            response = json.dumps(
                reflection,
                indent=2,
                ensure_ascii=False,
            )
        except Exception:  # noqa: F841
            reflection = {}
            response = self.prompt_dict["retry_hint"].format_map(
                {"state": "making the reflection"},
            )

        if reflection.get("rephrase_subtask", False) and reflection[
            "rephrase_subtask"
        ].get(
            "need_rephrase",
            False,
        ):  # type: ignore[index]
            self.current_subtask[-1].working_plan = reflection[
                "rephrase_subtask"
            ][
                "rephrased_plan"
            ]  # type: ignore[index]
        elif reflection.get("decompose_subtask", False) and reflection[
            "decompose_subtask"
        ].get(
            "need_decompose",
            False,
        ):  # type: ignore[index]
            if len(self.current_subtask) <= self.max_depth:
                intermediate_report = (
                    await self.summarize_intermediate_results()
                )
                self.current_subtask.append(
                    SubTaskItem(
                        objective=reflection[
                            "decompose_subtask"
                        ].get(  # type: ignore[index]
                            "failed_subtask",
                            None,
                        ),
                    ),
                )
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=response,
                        ),
                    ],
                    metadata={
                        "update_memory": True,
                        "intermediate_report": intermediate_report,
                    },
                )
            else:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=self.prompt_dict["max_depth_hint"],
                        ),
                    ],
                )
        else:
            pass
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=response,
                ),
            ],
        )

    # pylint: disable=invalid-overridden-method, unused-argument
    async def generate_response(  #
            self,
            response: str,
            **_kwargs: Any,
    ) -> ToolResponse:
        """Generate a detailed report as a response.

        Besides, when calling this function, the reasoning-acting memory will
        be cleared, so your response should contain a brief summary of what
        you have done so far.

        Args:
            response (`str`):
                Your response to the user.
        """
        checklist = self.current_subtask[0].knowledge_gaps
        completed_subtask = self.current_subtask.pop()

        if len(self.current_subtask) == 0:
            (
                summarized_content,
                _,
            ) = await self._generate_deepresearch_report(
                checklist=checklist,
            )
            response_msg = Msg(
                name=self.name,
                role="assistant",
                content=json.dumps(
                    summarized_content.content[0]["output"][0],
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Successfully generated detailed report.",
                    ),
                ],
                metadata={
                    "success": True,
                    "response_msg": response_msg,
                },
                is_last=True,
            )
        else:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=self.prompt_dict[
                            "subtask_complete_hint"
                        ].format_map(
                            {
                                "cur_obj": completed_subtask.objective,
                                "next_obj": self.current_subtask[-1].objective,
                            },
                        ),
                    ),
                ],
                metadata={
                    "success": True,
                },
                is_last=True,
            )
