import os
import json
import asyncio

from typing import Type, Optional, Any, Tuple, Literal
from datetime import datetime
from copy import deepcopy
import shortuuid
from agentscope.plan import PlanNotebook
from pydantic import BaseModel

from agentscope import logger, setup_logger
from agentscope.mcp import StatefulClientBase
from agentscope.agent import ReActAgent
from agentscope.model import ChatModelBase
from agentscope.formatter import FormatterBase
from agentscope.memory import MemoryBase, LongTermMemoryBase
from agentscope.tool import (
    ToolResponse,
    view_text_file,
    write_text_file, Toolkit,
)
from agentscope.message import (
    Msg,
    ToolUseBlock,
    TextBlock,
    ToolResultBlock,
)

from tut_agentscope.t10_deepresearch.deep_research_agent import SubTaskItem


class DeepResearchAgent(ReActAgent):
    def __init__(self, name: str,
                 sys_prompt: str,
                 model: ChatModelBase,
                 formatter: FormatterBase,
                 toolkit: Toolkit | None = None,
                 memory: MemoryBase | None = None,
                 long_term_memory: LongTermMemoryBase | None = None,
                 long_term_memory_mode: Literal[
                     "agent_control",
                     "static_control",
                     "both",
                 ] = "both",
                 enable_meta_tool: bool = False,
                 parallel_tool_calls: bool = False,
                 max_iters: int = 10,
                 plan_notebook: PlanNotebook | None = None,
                 print_hint_msg: bool = False) -> None:
        super().__init__(name, sys_prompt,
                         model, formatter,
                         toolkit, memory,
                         long_term_memory, long_term_memory_mode,
                         enable_meta_tool, parallel_tool_calls,
                         max_iters, plan_notebook, print_hint_msg)
        self.current_subtask = []

    @property
    def sys_prompt(self) -> str:
        return super().sys_prompt

    async def reply(self, msg: Msg | list[Msg] | None = None, structured_model: Type[BaseModel] | None = None) -> Msg:
        self.user_query = msg.get_text_content()
        self.current_subtask.append(
            SubTaskItem(objective=self.user_query),
        )
        return await super().reply(msg, structured_model)

    async def _reasoning(self) -> Msg:
        return await super()._reasoning()

    async def _acting(self, tool_call: ToolUseBlock) -> Msg | None:
        return await super()._acting(tool_call)

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        return await super().observe(msg)

    async def _summarizing(self) -> Msg:
        return await super()._summarizing()

    async def handle_interrupt(self, _msg: Msg | list[Msg] | None = None) -> Msg:
        return await super().handle_interrupt(_msg)

    def generate_response(self, response: str, **kwargs: Any) -> ToolResponse:
        return super().generate_response(response, **kwargs)
