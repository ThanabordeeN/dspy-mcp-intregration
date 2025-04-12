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
    MCP-powered ReAct agents with less boilerplate code. It can handle multiple
    MCP server sessions, allowing you to interact with different MCP servers
    simultaneously with a unified interface.
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
        
        # Dictionary to store multiple sessions
        self.sessions = {}
        self.active_session_id = None
        
        # Store all tools from all sessions
        self.all_tools = []
        self.tool_to_session = {}  # Maps tool name to session ID
        
        # Master react agent that uses all tools
        self.master_agent = None
    
    async def setup(self, server_configs=None, **kwargs):
        """
        Set up one or multiple MCP environments and create a unified ReAct agent.
        
        Args:
            server_configs: List of server configuration dictionaries, each containing:
                - command: Command to run the MCP server
                - args: Arguments for the command
                - session_id: Identifier for this session
                - env: (Optional) Environment variables for the process
            **kwargs: If server_configs is None, these are used for a single setup:
                - command: Command to run the MCP server
                - args: Arguments for the command
                - session_id: (Optional) Identifier for this session (default: "default")
                - env: (Optional) Environment variables for the process
                
        Returns:
            Self for method chaining
        """
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession, StdioServerParameters
        
        # Handle both single setup and multiple setup cases
        if server_configs is None:
            # Single setup using kwargs
            command = kwargs.get('command')
            args = kwargs.get('args')
            session_id = kwargs.get('session_id', 'default')
            env = kwargs.get('env')
            
            if command is None or args is None:
                raise ValueError("For single setup, both 'command' and 'args' are required")
                
            await self._setup_single_session(command, args, session_id, env)
        else:
            # Multiple setup using server_configs list
            if not isinstance(server_configs, list):
                raise ValueError("server_configs must be a list of server configuration dictionaries")
                
            for config in server_configs:
                command = config.get('command')
                args = config.get('args')
                session_id = config.get('session_id', f"session_{len(self.sessions)}")
                env = config.get('env')
                
                if command is None or args is None:
                    raise ValueError(f"For config {session_id}, both 'command' and 'args' are required")
                    
                await self._setup_single_session(command, args, session_id, env)
        
        # After all sessions are set up, create a master agent with all tools
        await self._create_master_agent()
                
        return self
    
    async def _setup_single_session(self, command, args, session_id="default", env=None):
        """
        Set up a single MCP environment and create a ReAct agent.
        
        Args:
            command: Command to run the MCP server
            args: Arguments for the command
            session_id: Identifier for this session (default: "default")
            env: Optional environment variables for the process
            
        Returns:
            Session container dictionary
        """
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession, StdioServerParameters
        
        # Set up the server parameters
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )
        
        # Create a session container
        session_container = {}
        
        # Store the context managers rather than their results
        session_container["stdio_context"] = stdio_client(server_params)
        # Connect to the MCP server using proper context handling
        session_container["read"], session_container["write"] = await session_container["stdio_context"].__aenter__()
        
        # Create session as a context manager and store it
        session_container["session_context"] = ClientSession(session_container["read"], session_container["write"])
        session_container["session"] = await session_container["session_context"].__aenter__()
        
        # Initialize the connection
        await session_container["session"].initialize()
        
        # Get tools for this session and track them
        client = MCPClient(session_container["session"])
        tools = await client.get_dspy_tools()
        session_container["tools"] = tools
        
        # Create a session-specific React agent
        session_container["react_agent"] = await create_mcp_react(
            session_container["session"], 
            self.signature,
            max_iters=self.max_iters
        )
        
        # Store this session
        self.sessions[session_id] = session_container
        
        # Map each tool to its session
        for tool in tools:
            # Store the original tool function
            original_func = tool.func
            tool_name = tool.name
            
            # Map tool to session
            self.tool_to_session[tool_name] = session_id
            
        # Set as active session if it's the first one or explicitly requested
        if self.active_session_id is None:
            self.active_session_id = session_id
        
        return session_container
    
    async def _create_master_agent(self):
        """
        Create a master ReAct agent that has access to all tools across all sessions.
        """
        if not self.sessions:
            raise ValueError("No sessions have been set up yet. Call setup() first.")
        
        # Collect all tools from all sessions
        all_tools = []
        for session_id, session_container in self.sessions.items():
            all_tools.extend(session_container["tools"])
            
        # Save all tools for reference
        self.all_tools = all_tools
        
        # Create a new master ReAct agent with all tools
        self.master_agent = dspy.ReAct(self.signature, all_tools, max_iters=self.max_iters)
    
    def set_active_session(self, session_id: str):
        """
        Set the active session by its identifier.
        
        Args:
            session_id: Identifier of the session to set as active
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session ID '{session_id}' does not exist")
        self.active_session_id = session_id
    
    def get_session_ids(self) -> List[str]:
        """
        Get a list of all session identifiers.
        
        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())
    
    async def run(self, request: str):
        """
        Run the master ReAct agent with the specified request.
        This automatically routes tool calls to the appropriate sessions.
        
        Args:
            request: The user's request to process
            
        Returns:
            The ReAct agent's response
        """
        if not self.master_agent:
            raise ValueError("The agent has not been set up. Call setup() first.")
            
        return await self.master_agent.async_forward(request=request)
    
    # Support context manager protocol for automatic cleanup
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def cleanup(self, session_id: Optional[str] = None):
        """
        Clean up resources for one or all sessions.
        
        Args:
            session_id: (Optional) Identifier of the session to clean up. If None, all sessions are cleaned up.
        """
        try:
            if session_id is None:
                # Clean up all sessions
                session_ids = list(self.sessions.keys())
                for sid in session_ids:
                    try:
                        await self._cleanup_single_session(sid)
                    except Exception as e:
                        print(f"Error cleaning up session {sid}: {e}")
                
                # Reset tool mappings and master agent
                self.tool_to_session = {}
                self.all_tools = []
                self.master_agent = None
                self.sessions = {}
            else:
                # Clean up a specific session
                if session_id in self.sessions:
                    await self._cleanup_single_session(session_id)
                    self.sessions.pop(session_id, None)
            
            # Additional cleanup with a slightly longer wait to ensure all tasks complete
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    async def _cleanup_single_session(self, session_id: str):
        """
        Clean up resources for a single session.
        
        Args:
            session_id: Identifier of the session to clean up
        """
        if session_id not in self.sessions:
            return
            
        session_container = self.sessions[session_id]
        
        # Cleanup session context first
        if "session_context" in session_container and session_container["session_context"]:
            try:
                await session_container["session_context"].__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing session context: {e}")
            session_container["session_context"] = None
            session_container["session"] = None
        
        # Then cleanup stdio context
        if "stdio_context" in session_container and session_container["stdio_context"]:
            try:
                await session_container["stdio_context"].__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing stdio context: {e}")
            session_container["stdio_context"] = None
            session_container["read"] = None
            session_container["write"] = None


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


async def cleanup_session(session=None, stdio_context=None):
    """
    Clean up resources related to MCP server connections.
    
    Args:
        session: An optional MCP session to clean up
        stdio_context: An optional stdio context to clean up
    """
    try:
        # Close session if provided
        if session:
            try:
                await session.close()
            except Exception as e:
                print(f"Error closing MCP session: {e}")
        
        # Close stdio context if provided
        if stdio_context:
            try:
                await stdio_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing stdio context: {e}")
        
        # Give asyncio loop time to process closures
        await asyncio.sleep(0.2)
    except Exception as e:
        print(f"Error during MCP resource cleanup: {e}")
