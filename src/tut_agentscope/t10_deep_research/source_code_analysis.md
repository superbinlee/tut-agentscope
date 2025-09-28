# Deep Research Agent 源码详细执行流程分析

## 1. 程序入口 [main.py](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\main.py)

### [main](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t2_mcp\tool_group.py#L15-L49) 函数执行流程
1. **初始化搜索客户端**
   - 创建 `StdIOStatefulClient` 实例 `tavily_search_client`
   - 配置客户端参数：
     - `name`: "tavily_mcp"
     - `command`: "npx"
     - `args`: ["-y", "tavily-mcp@latest"]
     - `env`: 设置 `TAVILY_API_KEY` 环境变量
   - 设置工作目录 `agent_working_dir`，默认为 `deepresearch_agent_demo_env`

2. **建立连接**
   - 调用 `await tavily_search_client.connect()` 建立与 Tavily 搜索服务的连接

3. **初始化代理**
   - 创建 [DeepResearchAgent](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L64-L1123) 实例 [agent](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t1_react\react_agent.py#L14-L21)，配置以下组件：
     - `name`: "Friday"
     - `sys_prompt`: "You are a helpful assistant named Friday."
     - `model`: `DashScopeChatModel` 使用 qwen-max 模型
     - `formatter`: `DashScopeChatFormatter`
     - [memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L155-L155): `InMemoryMemory`
     - `search_mcp_client`: 传入已连接的 `tavily_search_client`
     - [tmp_file_storage_dir](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L156-L156): 工作目录路径

4. **执行任务**
   - 构造用户消息 `Msg` 对象，包含查询内容
   - 调用 `await agent(msg)` 执行研究任务
   - 记录结果日志

5. **资源清理**
   - 无论成功与否，最终调用 `await tavily_search_client.close()` 关闭连接

## 2. 核心实现 [deep_research_agent.py](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py)

### [DeepResearchAgent.__init__](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L90-L184) 初始化流程
1. **提示词加载**
   - 调用 [load_prompt_dict()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\utils.py#L159-L325) 加载所有内置提示词
   - 增强系统提示词，添加工作说明和工具使用规则：
     - `add_note`: 来自 [prompt_worker_additional_sys_prompt.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_worker_additional_sys_prompt.md)
     - `tool_use_rule`: 来自 [prompt_tool_usage_rules.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_tool_usage_rules.md)

2. **工具注册**
   - 注册文件操作工具: `view_text_file`, `write_text_file`
   - 异步注册 MCP 客户端用于搜索功能
   - 注册内部工具函数:
     - [reflect_failure](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L948-L1054)
     - [summarize_intermediate_results](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L713-L840)

3. **属性设置**
   - 初始化参数：[max_depth](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L154-L154)(默认3), [tmp_file_storage_dir](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L156-L156)
   - 设置搜索相关函数名常量：
     - [search_function](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L166-L166): "tavily-search"
     - [extract_function](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L167-L167): "tavily-extract"
     - [read_file_function](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L168-L168): "view_text_file"
     - [write_file_function](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L169-L169): "write_text_file"
     - [summarize_function](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L170-L170): "summarize_intermediate_results"
   - 初始化内部状态变量：[intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172), [report_path_based](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L173-L175), [report_index](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L176-L176) 等

### [DeepResearchAgent.reply](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L186-L269) 主循环执行流程
1. **任务初始化**
   - 将用户查询保存到 [user_query](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L178-L178) 属性
   - 将查询添加到 [current_subtask](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L157-L157) 列表作为根任务
   - 调用 [decompose_and_expand_subtask()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L497-L564) 生成初始工作计划
   - 将知识缺口添加到用户消息内容中
   - 将用户消息添加到主内存 [memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L155-L155)

2. **主循环处理** (最多 `max_iters` 次)
   - **工作计划生成检查**
     - 如果当前子任务没有工作计划，则调用 [decompose_and_expand_subtask()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L497-L564)
   
   - **推理提示词构造**
     - 获取当前子任务目标 [objective](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L59-L59)
     - 获取当前工作计划 [working_plan](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L60-L60) 和知识缺口 [knowledge_gaps](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L61-L61)
     - 构造提示词包含：
       - 当前子任务
       - 工作计划
       - 知识缺口
       - 研究深度 (子任务列表长度)
     - 创建 `TextBlock` 消息并添加到 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172)
   
   - **执行推理**
     - 备份当前主内存状态
     - 将 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172) 添加到主内存
     - 调用 `_reasoning()` 生成模型响应
     - 恢复主内存状态
   
   - **工具调用处理**
     - 遍历响应中的所有 `tool_use` 块
     - 将工具调用添加到 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172)
     - 调用 [_acting(tool_call)](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L271-L408) 执行具体工具
     - 根据工具执行结果决定是否返回最终响应

3. **迭代结束处理**
   - 达到最大迭代次数后调用 [_summarizing()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L928-L946) 生成总结报告

### [DeepResearchAgent._acting](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L271-L408) 工具执行流程
1. **工具调用执行**
   - 创建工具结果消息 `tool_res_msg`
   - 调用 `self.toolkit.call_tool_function(tool_call)` 执行具体工具
   - 处理流式响应数据

2. **特殊工具处理**
   - **完成函数处理** (`finish_function_name`):
     - 如果执行成功且在根层级，调用 [generate_response()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L1057-L1123) 生成最终响应
     - 清空 [current_subtask](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L157-L157) 并返回响应消息
   
   - **总结函数处理** ([summarize_intermediate_results](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L713-L840)):
     - 清空 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172)
     - 将总结内容添加到主内存 [memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L155-L155)
   
   - **搜索结果处理** (`tavily-search`, `tavily-extract`):
     - 调用 [truncate_search_result()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\utils.py#L55-L69) 截断过长结果
     - 更新工具结果消息内容

3. **深度搜索处理**
   - 如果是搜索工具调用，调用 [_follow_up()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L566-L711) 进行深度信息挖掘
   - 根据需要更新内存状态

4. **内存更新**
   - 将工具结果消息添加到 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172)
   - 如需要更新主内存，则清空 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172) 并添加到 [memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L155-L155)

### [DeepResearchAgent.decompose_and_expand_subtask](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L497-L564) 任务分解流程
1. **深度检查**
   - 检查 `len(current_subtask)` 是否超过 [max_depth](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L154-L154) 限制

2. **提示词构造**
   - 系统提示词: 使用 [prompt_decompose_subtask.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_decompose_subtask.md) 内容
   - 用户提示词包含:
     - 历史计划: 遍历 [current_subtask](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L157-L157) 收集之前的工作计划
     - 当前任务目标: `current_subtask[-1].objective`

3. **模型调用**
   - 调用 [get_model_output()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L410-L450) 方法
   - 使用 [SubtasksDecomposition](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L5-L28) 模型格式化输出
   - 输入参数包含任务描述和历史信息

4. **结果更新**
   - 解析模型输出，更新当前子任务的:
     - [knowledge_gaps](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L61-L61): 知识缺口清单
     - [working_plan](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L60-L60): 工作计划步骤
   - 返回结构化结果

### [DeepResearchAgent._follow_up](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L566-L711) 深度搜索流程
1. **深度检查**
   - 检查 `len(current_subtask)` 是否小于 [max_depth](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L154-L154) 允许进一步扩展

2. **信息评估**
   - 系统提示词: 使用 [prompt_deeper_expansion.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_deeper_expansion.md) 内容
   - 用户提示词包含:
     - 当前搜索目标: `tool_call["input"].get("query", "")`
     - 原始任务知识缺口: `current_subtask[0].knowledge_gaps`
     - 当前工作计划: `current_subtask[-1].working_plan`
     - 搜索结果: `search_results`
   - 调用 [get_model_output()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L410-L450) 使用 [WebExtraction](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L31-L58) 模型

3. **网页提取决策**
   - 检查 [need_more_information](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L41-L43) 是否为 `True`
   - 如果需要:
     - 提取目标 URL: `follow_up_subtask.get("url", None)`
     - 调用网页提取工具 `tavily-extract` 获取详细内容
     - 截断提取结果

4. **充分性判断**
   - 系统提示词: 使用 `follow_up_judge_sys_prompt`
   - 输入上下文包括搜索、提取全过程
   - 调用 [get_model_output()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L410-L450) 使用 [FollowupJudge](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L61-L73) 模型
   - 如果信息不足:
     - 调用 [summarize_intermediate_results()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L713-L840) 生成中间报告
     - 创建新的子任务 [SubTaskItem](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L56-L61) 添加到 [current_subtask](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L157-L157)
     - 返回需要更新内存的工具响应

### [DeepResearchAgent.summarize_intermediate_results](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L713-L840) 中间结果总结流程
1. **内容检查**
   - 检查 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172) 是否为空
   - 如果为空返回无结果提示

2. **工作计划更新**
   - 如果是主动调用(最后一个消息是总结函数调用):
     - 使用提示词 `summarize_hint` 更新工作计划状态
     - 标记已完成的步骤为 `[DONE]`

3. **报告生成**
   - 系统提示词: 使用 [prompt_inprocess_report.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_inprocess_report.md) 内容，添加报告前缀
   - 用户提示词包含:
     - 原始任务目标: `current_subtask[0].objective`
     - 知识缺口清单: `current_subtask[0].knowledge_gaps`
     - 当前工作计划: `current_subtask[-1].working_plan`
     - 工具执行结果: 遍历 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172) 收集所有工具结果
   - 调用模型生成中间报告

4. **文件保存**
   - 构造文件路径: `{report_path_based}_inprocess_report_{report_index}.md`
   - 调用 `write_text_file` 工具保存报告
   - 递增 [report_index](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L176-L176)
   - 返回保存结果提示

### [DeepResearchAgent._generate_deepresearch_report](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L842-L926) 最终报告生成流程
1. **中间报告收集**
   - 检查 [report_index](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L176-L176) 是否大于1
   - 如果有中间报告:
     - 遍历所有中间报告文件
     - 调用 `view_text_file` 工具读取内容
     - 合并成完整草稿报告
   - 如果没有中间报告:
     - 使用 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172) 作为输入

2. **最终报告生成**
   - 系统提示词: 使用 [prompt_deepresearch_summary_report.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_deepresearch_summary_report.md) 内容
     - 替换 `{original_task}` 为用户原始查询
     - 替换 `{checklist}` 为知识缺口清单
   - 用户提示词: 提供草稿报告内容
   - 调用模型生成最终详细研究报告

3. **文件保存**
   - 构造文件路径: `{report_path_based}_detailed_report.md`
   - 调用 `write_text_file` 工具保存最终报告

### [DeepResearchAgent.reflect_failure](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L948-L1054) 失败反思流程
1. **历史记录分析**
   - 收集 [intermediate_memory](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L172-L172) 中的对话历史
   - 系统提示词: 使用 [prompt_reflect_failure.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_reflect_failure.md) 内容
   - 用户提示词包含:
     - 对话历史: JSON 格式的完整对话记录
     - 当前工作计划: `current_subtask[-1].working_plan`

2. **失败原因分析**
   - 调用 [get_model_output()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L410-L450) 使用 [ReflectFailure](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L76-L139) 模型
   - 分析失败的子任务和原因

3. **策略调整**
   - **重新表述任务**:
     - 如果 `rephrase_subtask.need_rephrase` 为 `True`
     - 更新当前工作计划中的问题步骤
   - **进一步分解**:
     - 如果 `decompose_subtask.need_decompose` 为 `True`
     - 检查深度限制
     - 调用 [summarize_intermediate_results()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L713-L840) 生成中间报告
     - 创建新的子任务 [SubTaskItem](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L56-L61) 添加到 [current_subtask](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L157-L157)
     - 返回需要更新内存的工具响应

## 3. 工具函数 [utils.py](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\utils.py) 执行流程

### [load_prompt_dict](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\utils.py#L159-L325) 提示词加载流程
1. **文件读取**
   - 遍历 `built_in_prompt` 目录下的所有提示词文件
   - 读取以下文件内容:
     - [prompt_worker_additional_sys_prompt.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_worker_additional_sys_prompt.md) → `add_note`
     - [prompt_tool_usage_rules.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_tool_usage_rules.md) → `tool_use_rule`
     - [prompt_decompose_subtask.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_decompose_subtask.md) → `decompose_sys_prompt`
     - [prompt_deeper_expansion.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_deeper_expansion.md) → `expansion_sys_prompt`
     - [prompt_inprocess_report.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_inprocess_report.md) → `summarize_sys_prompt`
     - [prompt_deepresearch_summary_report.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_deepresearch_summary_report.md) → `reporting_sys_prompt`
     - [prompt_reflect_failure.md](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\prompt_reflect_failure.md) → `reflect_sys_prompt`

2. **模板字符串加载**
   - 加载各种处理阶段的提示词模板:
     - `reasoning_prompt`: 推理阶段模板
     - `previous_plan_inst`: 历史计划模板
     - `max_depth_hint`: 深度限制提示
     - `expansion_inst`: 扩展搜索模板
     - 等等...

3. **字典构建**
   - 将所有读取的提示词内容按功能分类存储在字典中
   - 为不同处理阶段提供对应的提示词模板

### [truncate_search_result](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\utils.py#L55-L69) 搜索结果截断流程
1. **参数验证**
   - 验证搜索和提取函数名是否为默认值
   - 如果不是则抛出 `NotImplementedError`

2. **文本截断**
   - 遍历搜索结果列表中的每条记录
   - 对每条记录的 `text` 字段调用 [truncate_by_words()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\utils.py#L27-L52)
   - 限制文本长度为 [TOOL_RESULTS_MAX_WORDS](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\utils.py#L11-L11)(5000词)

### [get_model_output](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L410-L450) 模型调用流程
1. **格式化消息**
   - 调用 `formatter.format()` 格式化输入消息

2. **工具调用配置**
   - 如果提供 `format_template`，调用 [get_dynamic_tool_call_json()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\utils.py#L94-L123) 生成工具调用JSON

3. **模型调用**
   - 调用 `model()` 方法执行推理
   - 处理流式或非流式响应

4. **结果处理**
   - 如果使用结构化输出，调用 [get_structure_output()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\utils.py#L126-L156) 提取结果
   - 返回最终输出

## 4. 数据模型 `built_in_prompt/promptmodule.py`

### [SubtasksDecomposition](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L5-L28) 模型
- [knowledge_gaps](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L61-L61): 字符串类型，描述必需的知识缺口和可选的视角扩展缺口
- [working_plan](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L60-L60): 字符串类型，描述逻辑有序的3-5步工作计划

### [WebExtraction](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L31-L58) 模型
- [reasoning](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L66-L70): 字符串类型，决策理由说明
- [need_more_information](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L41-L43): 布尔类型，是否需要更多信息
- [title](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L44-L48): 字符串类型，需要提取的搜索结果标题
- [url](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L49-L53): 字符串类型，需要提取的搜索结果URL
- [subtask](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L54-L58): 字符串类型，后续获取信息的任务描述

### [FollowupJudge](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L61-L73) 模型
- [reasoning](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L66-L70): 字符串类型，判断理由说明
- [is_sufficient](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L71-L73): 布尔类型，信息是否充足

### [ReflectFailure](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L76-L139) 模型
- [rephrase_subtask](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L81-L111): 字典类型，包含是否需要重新表述任务的信息
- [decompose_subtask](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\built_in_prompt\promptmodule.py#L112-L139): 字典类型，包含是否需要分解任务的信息

## 5. 整体执行顺序

1. **程序启动** → [main()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\main.py#L15-L64) 函数执行
   - 初始化搜索客户端和工作环境
   - 创建 [DeepResearchAgent](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L64-L1123) 实例

2. **代理初始化** → [DeepResearchAgent.__init__()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L90-L184)
   - 加载所有提示词模板
   - 注册工具函数
   - 设置初始状态

3. **任务接收** → 用户查询通过 `Msg` 传递给代理

4. **任务分解** → [decompose_and_expand_subtask()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L497-L564) 生成初始工作计划
   - 分析任务识别知识缺口
   - 制定3-5步工作计划

5. **主循环执行** → [reply()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L186-L269) 主循环中:
   - **推理阶段**:
     - 构造包含任务、计划、知识缺口的提示词
     - 调用 `_reasoning()` 生成模型响应
   - **执行阶段**:
     - 处理工具调用 [_acting()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L271-L408)
     - 必要时进行深度搜索 [_follow_up()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L566-L711)
     - 完成步骤后总结 [summarize_intermediate_results()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L713-L840)

6. **异常处理** → [reflect_failure()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L948-L1054) 处理失败情况
   - 分析失败原因
   - 调整策略(重新表述或进一步分解)

7. **报告生成** → [_generate_deepresearch_report()](file://D:\root\projects\python\tut-agentscope\src\tut_agentscope\t10_deep_research\deep_research_agent.py#L842-L926) 生成最终报告
   - 收集所有中间报告
   - 生成详细研究报告

8. **程序结束** → 返回最终结果并清理资源