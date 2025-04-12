"""
Enhanced resource management for Model Context Protocol (MCP) in DSPy.

This module provides robust utilities for managing MCP server connections,
including support for multiple concurrent sessions with proper resource handling.
"""

import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from dspy.primitives.tool import Tool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dspy.mcp")


class MCPTool(Tool):
    """Wrapper for MCP tools with server attribution for routing."""
    
    def __init__(self, manager, server_name: str, tool_info: Dict[str, Any]):
        """
        Create a Tool from an MCP tool description.
        
        Args:
            manager: The MCPManager instance
            server_name: Name of the server this tool belongs to
            tool_info: Tool information from session.list_tools()
        """
        self.manager = manager
        self.server_name = server_name
        
        # Extract necessary information from tool_info
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
            
        # Add server name to tool name for better error messages
        # desc = f"[{server_name}] {desc}"
        
        # Map JSON schema to tool arguments
        args, arg_types, arg_desc = {}, {}, {}
        schema = None
        
        if hasattr(tool_info, 'inputSchema'):
            schema = tool_info.inputSchema
        elif isinstance(tool_info, dict) and "inputSchema" in tool_info:
            schema = tool_info["inputSchema"]
            
        if schema:
            args, arg_types, arg_desc = map_json_schema_to_tool_args(schema)
        
        # Define the function that routes tool calls to the appropriate server
        async def call_tool_async(**kwargs):
            # Call the tool on the specified server
            server = self.manager.servers.get(self.server_name)
            if not server:
                raise ValueError(f"Server {self.server_name} not found")
                
            result = await server.execute_tool(name, kwargs)
            
            # Process results based on their structure
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and result.content:
                    try:
                        # Try to extract text content
                        text_content = [item.text for item in result.content if hasattr(item, 'text')]
                        return "\n".join(filter(None, text_content))
                    except Exception:
                        # Fall back to string representation
                        return str(result.content)
                else:
                    return str(result.content)
            elif hasattr(result, 'text'):
                return result.text
            elif isinstance(result, dict):
                # For dictionary results, handle common patterns
                if "content" in result:
                    return str(result["content"])
                elif "result" in result:
                    return str(result["result"])
            
            # Default fallback
            return "Tool executed successfully, but returned no parseable content."

        # Wrapper that returns the coroutine
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

class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        Initialize a server connection manager.
        
        Args:
            name: Name identifier for this server
            config: Server configuration dictionary with command, args, and optional env
        """
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """
        Initialize the server connection.
        
        Raises:
            ValueError: If command is None or invalid
            Exception: If server initialization fails
        """
        command = (
            shutil.which("npx") 
            if self.config["command"] == "npx" 
            else self.config["command"]
        )
        
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self.session.initialize()
        except Exception as e:
            logger.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """
        List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        return tools_response.tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """
        Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            retries: Number of retry attempts
            delay: Delay between retries in seconds

        Returns:
            Tool execution result

        Raises:
            RuntimeError: If server is not initialized
            Exception: If tool execution fails after all retries
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logger.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                if self.exit_stack is not None:
                    await self.exit_stack.aclose()
                    # Clear references after closing
                    self.session = None
                    # Add a small delay to allow task completion
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error during cleanup of server {self.name}: {e}")
            finally:
                # Reset to a fresh exit stack
                self.exit_stack = AsyncExitStack()


class MCPServerManager:
    """
    Manages multiple MCP server connections for DSPy.
    
    This class provides a more robust interface for creating and managing 
    multiple MCP server sessions with proper resource handling.
    """
    
    def __init__(self):
        """Initialize the MCP server manager."""
        self.servers: Dict[str, Server] = {}
        self._exit_stack = AsyncExitStack()
    
    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """
        Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            json.JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)
    
    async def initialize_servers(self, config: Dict[str, Any]) -> None:
        """
        Initialize all servers defined in the config.
        
        Args:
            config: Server configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        if "mcpServers" not in config:
            raise ValueError("Configuration must contain 'mcpServers' section")
        
        for name, server_config in config["mcpServers"].items():
            server = Server(name, server_config)
            await server.initialize()
            self.servers[name] = server
    
    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools from all servers.
        
        Returns:
            List of all tools with server attribution
        """
        all_tools = []
        for name, server in self.servers.items():
            tools = await server.list_tools()
            for tool in tools:
                # Add server attribution
                tool_info = tool
                if hasattr(tool_info, "server"):
                    tool_info.server = name
                elif isinstance(tool_info, dict):
                    tool_info["server"] = name
                all_tools.append(tool_info)
        
        return all_tools
    
    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on a specific server.
        
        Args:
            server_name: Name of the server to use
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If server doesn't exist
        """
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not found")
            
        server = self.servers[server_name]
        return await server.execute_tool(tool_name, arguments)
    
    async def cleanup(self) -> None:
        """Clean up all server resources."""
        cleanup_tasks = []
        for server in self.servers.values():
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        self.servers = {}
    
    async def __aenter__(self) -> 'MCPServerManager':
        """Support use as an async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources when exiting context."""
        await self.cleanup()

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