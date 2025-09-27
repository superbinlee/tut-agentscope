import asyncio
from typing import List, Dict, Any
from fastmcp.client import Client


async def get_mcp_tools_info(url: str) -> List[Dict[str, Any]]:
    """异步获取工具信息的内部函数"""
    try:
        async with Client(url) as client:
            # 检查连接状态
            if not client.is_connected():
                raise ConnectionError(f"无法连接到MCP服务: {url}")
            # 获取工具列表
            tools = await client.list_tools()
            # 组装工具信息字典
            tools_info = []
            for tool in tools:
                tool_dict = {
                    "name": getattr(tool, "name", None),
                    "title": getattr(tool, "title", None),
                    "description": getattr(tool, "description", None),
                    "inputSchema": getattr(tool, "inputSchema", None),
                    "outputSchema": getattr(tool, "outputSchema", None)
                }
                # 过滤掉None值，保持数据整洁
                tool_dict = {k: v for k, v in tool_dict.items() if v is not None}
                tools_info.append(tool_dict)
            return tools_info
    except Exception as e:
        raise RuntimeError(f"获取MCP工具信息时发生错误: {str(e)}")


async def main():
    # 基本使用
    mcp_url = "http://localhost:9121/mcp"
    try:
        # 获取工具信息
        tools_info = await get_mcp_tools_info(mcp_url)
        print("工具信息:", tools_info)
    except Exception as e:
        print("错误:", e)


if __name__ == "__main__":
    asyncio.run(main())
