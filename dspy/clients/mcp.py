"""
Model Context Protocol (MCP) integration for DSPy.
This module provides utilities for integrating MCP tools with DSPy.
"""

import inspect
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Callable, Type, Union

import dspy
from dspy.primitives.tool import Tool


def map_json_schema_to_tool_args(schema: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    """
    Maps a JSON schema to tool arguments compatible with DSPy Tool.

    Args:
        schema: JSON schema describing tool arguments.

    Returns:
        A tuple of (args, arg_types, arg_desc) for the Tool constructor.
    """
    args = {}
    arg_types = {}
    arg_desc = {}

    if "properties" in schema:
        for name, prop in schema["properties"].items():
            args[name] = prop
            # We use Any as a fallback type since we don't have precise type information
            arg_types[name] = Any
            if "description" in prop:
                arg_desc[name] = prop["description"]

    return args, arg_types, arg_desc


class MCPTool(Tool):
    """
    Wrapper for MCP tools to be used with DSPy's ReAct.
    """
    def __init__(self, session, tool_info: Dict[str, Any]):
        """
        Creates a Tool from an MCP tool description.

        Args:
            session: MCP client session
            tool_info: Tool information from session.list_tools()
        """
        self.session = session
        self.tool_info = tool_info
        
        # Extract necessary information from tool_info
        # Handle both dictionary and object access patterns
        if hasattr(tool_info, 'name'):
            name = tool_info.name
        elif isinstance(tool_info, dict):
            name = tool_info.get("name", "unknown_tool")
        else:
            name = "unknown_tool"
            
        if hasattr(tool_info, 'description'):
            desc = tool_info.description
        elif isinstance(tool_info, dict):
            desc = tool_info.get("description", "")
        else:
            desc = ""
        
        # Map JSON schema to tool arguments
        args, arg_types, arg_desc = {}, {}, {}
        schema = None
        
        if hasattr(tool_info, 'inputSchema'):
            schema = tool_info.inputSchema
        elif isinstance(tool_info, dict) and "inputSchema" in tool_info:
            schema = tool_info["inputSchema"]
            
        if schema:
            args, arg_types, arg_desc = map_json_schema_to_tool_args(schema)
        
        # Define the async function to call the MCP tool
        async def call_tool_async(**kwargs):
            result = await session.call_tool(name, arguments=kwargs)
            # Ensure result content is serializable (e.g., extract text)
            if result and result.content:
                # Assuming content is a list of TextContent or similar
                # Adjust this based on the actual structure of result.content
                if isinstance(result.content, list) and len(result.content) > 0:
                     # Attempt to extract text, handle potential errors or different structures
                    try:
                        # Example: Join text from TextContent objects
                        text_content = [item.text for item in result.content if hasattr(item, 'text')]
                        session.clear()
                        return "\n".join(filter(None, text_content))
                        
                    except Exception:
                        # Fallback if content structure is different
                        return str(result.content) 
                else:
                    return str(result.content)
            return "Tool executed successfully, but returned no content."

        # The wrapper now directly returns the coroutine
        # It assumes ReAct will handle awaiting it correctly within the main event loop
        def call_tool_wrapper(**kwargs):
            return call_tool_async(**kwargs)
            
        super().__init__(
            func=call_tool_wrapper,
            name=name,
            desc=desc,
            args=args,
            arg_types=arg_types,
            arg_desc=arg_desc
        )


class MCPClient:
    """
    Client for working with Model Context Protocol (MCP) servers in DSPy.
    """
    def __init__(self, session):
        """
        Initialize the MCP client with a session.

        Args:
            session: An initialized MCP session
        """
        self.session = session
        self._tools_cache = None
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lists available tools from the MCP server.

        Returns:
            List of tool information dictionaries
        """
        if self._tools_cache is None:
            tools = await self.session.list_tools()
            self._tools_cache = tools.tools
        return self._tools_cache
    
    async def get_dspy_tools(self) -> List[Tool]:
        """
        Gets all available MCP tools as DSPy Tool objects.

        Returns:
            List of DSPy Tool objects
        """
        tools = await self.list_tools()
        return [MCPTool(self.session, tool) for tool in tools]


class MCPReactAgent:
    """
    A simplified wrapper for using MCP with ReAct in DSPy.
    
    This class provides a more user-friendly interface for creating and using
    MCP-powered ReAct agents with less boilerplate code.
    """
    
    def __init__(self, signature = None, max_iters: int = 10):
        """
        Initialize an MCP ReAct agent.
        
        Args:
            signature: The signature class for the ReAct module. If None, a default signature will be used.
            max_iters: Maximum number of iterations for ReAct reasoning
        """
        self.signature = signature 
        self.max_iters = max_iters
        self.react_agent = None
        self.session = None
        self._read = None
        self._write = None
        
    
    async def setup(self, 
                   command: str, 
                   args: List[str]):
        """
        Set up the MCP environment and create the ReAct agent.
        
        Args:
            command: Command to run the MCP server
            args: Arguments for the command
            
        Returns:
            Self for method chaining
        """
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession, StdioServerParameters
        
        # Set up the server parameters
        server_params = StdioServerParameters(
            command=command,
            args=args
        )
        
        # Store the context managers rather than their results
        self._stdio_context = stdio_client(server_params)
        # Connect to the MCP server using proper context handling
        self._read, self._write = await self._stdio_context.__aenter__()
        
        # Create session as a context manager and store it
        self._session_context = ClientSession(self._read, self._write)
        self.session = await self._session_context.__aenter__()
        
        # Initialize the connection
        await self.session.initialize()
        
        # Create the ReAct agent
        self.react_agent = await create_mcp_react(
            self.session, 
            self.signature,
            max_iters=self.max_iters
        )
        
        return self
    
    async def run(self, request: str):
        """
        Run the ReAct agent with the specified request.
        
        Args:
            request: The user's request to process
            
        Returns:
            The ReAct agent's response
        """
        if not self.react_agent:
            raise ValueError("The agent has not been set up. Call setup() first.")
            
        return await self.react_agent.async_forward(request=request)
    
    # Support context manager protocol for automatic cleanup
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, '_session_context') and self._session_context:
            await self._session_context.__aexit__(None, None, None)
            self.session = None
            
        if hasattr(self, '_stdio_context') and self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
            self._read = None
            self._write = None
        
        # Additional cleanup with a slightly longer wait to ensure all tasks complete
        await asyncio.sleep(0.2)


async def create_mcp_react(session, signature, max_iters=5):
    """
    Creates a DSPy ReAct module with MCP tools.

    Args:
        session: An initialized MCP session
        signature: The signature for the ReAct module
        max_iters: Maximum number of iterations for ReAct

    Returns:
        An initialized ReAct module with MCP tools
    """
    client = MCPClient(session)
    tools = await client.get_dspy_tools()
    return dspy.ReAct(signature, tools, max_iters=max_iters)


async def cleanup_session():
    """Clean up resources"""
    await asyncio.sleep(0.1) # Give asyncio loop time to process closures
