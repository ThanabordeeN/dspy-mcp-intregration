"""
Enhanced resource management for Model Context Protocol (MCP) in DSPy.

This module provides robust utilities for managing MCP server connections,
including support for multiple concurrent sessions with proper resource handling,
and integrates MCP tools seamlessly into the DSPy framework.
"""

import asyncio
import gc
import json
import logging
import os
import shutil
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import DSPy tools
from dspy.primitives.tool import Tool

#---------------------------------------------------------------------
# Constants and Configuration
#---------------------------------------------------------------------

# Default logging level - can be overridden
DEFAULT_LOG_LEVEL = logging.WARNING  # Less verbose default

#---------------------------------------------------------------------
# Logging Configuration
#---------------------------------------------------------------------

# Configure logger - initially null, configured by setup functions
logger = logging.getLogger("dspy.mcp")
logger.addHandler(logging.NullHandler())  # Prevent "no handler" warnings

def setup_logging(log_level=logging.INFO, log_to_file=False, log_dir=None):
    """Configure logging with optional file output and cleanup."""
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        if not log_dir:
            log_dir = Path.home() / "dspy_logs"
        else:
            log_dir = Path(log_dir)
            
        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean old log files (keep last 10)
        clean_old_logs(log_dir)
        
        # Create new log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"dspy_mcp_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
        
    return root_logger

def clean_old_logs(log_dir, keep_last=10):
    """Clean old log files, keeping only the specified number of latest files."""
    log_files = list(log_dir.glob("dspy_mcp_*.log"))
    if len(log_files) > keep_last:
        # Sort by modification time (oldest first)
        log_files.sort(key=lambda f: f.stat().st_mtime)
        # Delete oldest files
        for f in log_files[:-keep_last]:
            try:
                f.unlink()
            except (PermissionError, OSError):
                pass  # Skip if file is locked or cannot be deleted

def disable_logging():
    """Completely disable all logging from the MCP module."""
    # Get the logger
    mcp_logger = logging.getLogger("dspy.mcp")
    
    # Remove all existing handlers
    for handler in mcp_logger.handlers[:]:
        mcp_logger.removeHandler(handler)
    
    # Set level higher than CRITICAL to silence all messages
    mcp_logger.setLevel(logging.CRITICAL + 1)
    
    # Also handle the root logger to be sure
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL + 1)
    
    # Add a null handler to prevent "No handler found" warnings
    mcp_logger.addHandler(logging.NullHandler())
    
    return mcp_logger

# Default setup - can be modified by calling setup_logging() later
setup_logging(log_level=logging.INFO)

#---------------------------------------------------------------------
# Schema Utilities
#---------------------------------------------------------------------

def map_json_schema_to_tool_args(
    schema: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Type], Dict[str, str]]:
    """
    Maps a JSON schema to tool arguments compatible with DSPy Tool.

    Args:
        schema: JSON schema describing tool arguments, or None.

    Returns:
        A tuple of (args, arg_types, arg_desc) for the Tool constructor.
    """
    args: Dict[str, Any] = {}
    arg_types: Dict[str, Type] = {}
    arg_desc: Dict[str, str] = {}

    if schema and "properties" in schema:
        for name, prop in schema["properties"].items():
            args[name] = prop

            # Basic type mapping
            prop_type = prop.get("type", "string")
            if prop_type == "string":
                arg_types[name] = str
            elif prop_type == "integer":
                arg_types[name] = int
            elif prop_type == "number":
                arg_types[name] = float
            elif prop_type == "boolean":
                arg_types[name] = bool
            elif prop_type == "array":
                arg_types[name] = list
            elif prop_type == "object":
                arg_types[name] = dict
            else:
                arg_types[name] = Any  # Fallback for complex or unknown types

            arg_desc[name] = prop.get("description", "No description provided.")

            # Mark required fields in description
            if name in schema.get("required", []):
                arg_desc[name] += " (Required)"

    return args, arg_types, arg_desc

#---------------------------------------------------------------------
# Core Classes
#---------------------------------------------------------------------

class Server:
    """Manages a single MCP server connection and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        Initialize a server connection manager.

        Args:
            name: Name identifier for this server.
            config: Server configuration dictionary with command, args, and optional env.
        """
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._is_initialized: bool = False
        logger.info(f"Server instance '{self.name}' created.")

    async def initialize(self) -> None:
        """Initialize the server connection using an AsyncExitStack for resource management."""
        if self._is_initialized:
            logger.warning(f"Server '{self.name}' already initialized.")
            return
            
        if not self.exit_stack:
            self.exit_stack = AsyncExitStack()

        logger.info(f"Initializing server '{self.name}'...")
        
        # Get and validate command
        command_path = self._resolve_command_path()
        server_params = self._create_server_params(command_path)

        try:
            # Setup transport and session
            await self._setup_transport_and_session()
            self._is_initialized = True
            logger.info(f"Server '{self.name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing server '{self.name}': {e}", exc_info=True)
            await self.cleanup()
            raise

    def _resolve_command_path(self) -> str:
        """Resolve the command path for the server executable."""
        command_name = self.config.get("command")
        if not command_name:
            raise ValueError(f"Missing 'command' in config for server '{self.name}'")

        # Resolve command path
        command_path = (
            shutil.which("npx")
            if command_name == "npx"
            else shutil.which(command_name)
        )
        
        # If not found directly, assume it might be a relative/absolute path
        if command_path is None:
            command_path = command_name

        if not command_path:
            raise ValueError(f"Command '{command_name}' not found or invalid for server '{self.name}'.")

        logger.info(f"Resolved command for server '{self.name}': {command_path}")
        return command_path

    def _create_server_params(self, command_path: str) -> StdioServerParameters:
        """Create server parameters for the MCP connection."""
        return StdioServerParameters(
            command=command_path,
            args=self.config.get("args", []),
            env={**os.environ, **self.config.get("env", {})}
        )

    async def _setup_transport_and_session(self) -> None:
        """Set up the transport and client session."""
        command_path = self._resolve_command_path()
        server_params = self._create_server_params(command_path)
        
        # Enter the stdio_client context using the exit stack
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport

        # Enter the ClientSession context using the same exit stack
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        # Initialize the MCP session
        await self.session.initialize()

    async def list_tools(self) -> List[Any]:
        """List available tools from the server."""
        if not self.session or not self._is_initialized:
            raise RuntimeError(f"Server '{self.name}' is not initialized or session is not available.")

        logger.debug(f"Listing tools for server '{self.name}'...")
        tools_response = await self.session.list_tools()
        logger.debug(f"Received {len(tools_response.tools)} tools from server '{self.name}'.")
        return tools_response.tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 1,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool on the server with a retry mechanism."""
        self._validate_session()
        
        attempt = 0
        last_exception = None
        
        while attempt <= retries:
            try:
                logger.info(f"Executing tool '{tool_name}' on server '{self.name}' (Attempt {attempt + 1}/{retries + 1})...")
                logger.debug(f"Arguments: {arguments}")
                
                result = await self.session.call_tool(tool_name, arguments)
                
                logger.info(f"Tool '{tool_name}' executed successfully on server '{self.name}'.")
                return result
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Error executing tool '{tool_name}' on server '{self.name}' (Attempt {attempt + 1}): {e}"
                )
                
                attempt += 1
                if attempt <= retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries ({retries}) reached for tool '{tool_name}'. Failing.")
                    raise last_exception

        # Should not be reachable if retries >= 0, but added for safety
        raise RuntimeError(f"Tool execution failed unexpectedly for '{tool_name}' after retries.")

    def _validate_session(self) -> None:
        """Validate that the session is initialized and available."""
        if not self.session or not self._is_initialized:
            raise RuntimeError(f"Server '{self.name}' is not initialized or session is not available.")

    async def cleanup(self) -> None:
        """Clean up server resources managed by the AsyncExitStack."""
        if not self._cleanup_lock.locked():
            async with self._cleanup_lock:
                if self.exit_stack:
                    logger.info(f"Cleaning up resources for server '{self.name}'...")
                    try:
                        await self._shutdown_session()
                        await self.exit_stack.aclose()
                        logger.info(f"Resources for server '{self.name}' cleaned up.")
                    except RuntimeError as e:
                        self._handle_cleanup_error(e)
                    except Exception as e:
                        logger.error(f"Error during cleanup of server '{self.name}': {e}")
                        self._force_close_subprocesses()
                    finally:
                        self._reset_state()
                else:
                    logger.debug(f"Cleanup called for server '{self.name}', but no active exit stack found.")

    async def _shutdown_session(self) -> None:
        """Properly shutdown the session if available."""
        if self.session and hasattr(self.session, 'shutdown') and callable(self.session.shutdown):
            try:
                await self.session.shutdown()
                logger.debug(f"Session for server '{self.name}' shut down.")
            except Exception as e:
                logger.debug(f"Error shutting down session for server '{self.name}': {e}")

    def _handle_cleanup_error(self, error: Exception) -> None:
        """Handle specific cleanup errors."""
        if isinstance(error, RuntimeError) and "Attempted to exit cancel scope in a different task" in str(error):
            logger.warning(f"Task context error during cleanup of server '{self.name}'. "
                           f"Resources may still be cleaned up by Python's GC.")
            self._force_close_subprocesses()
        else:
            logger.error(f"Error during cleanup of server '{self.name}': {error}")
            self._force_close_subprocesses()

    def _reset_state(self) -> None:
        """Reset internal state after cleanup."""
        self.session = None
        self._is_initialized = False
        self.exit_stack = AsyncExitStack()
        self._force_gc_cleanup()

    def _force_close_subprocesses(self) -> None:
        """Force close any subprocesses that may have been created by this server."""
        import platform
        if platform.system() == 'Windows':
            try:
                import psutil
                current_process = psutil.Process()
                for child in current_process.children(recursive=True):
                    cmd_line = " ".join(child.cmdline()).lower()
                    if self.name.lower() in cmd_line or (
                        'command' in self.config and 
                        self.config['command'].lower() in cmd_line
                    ):
                        try:
                            logger.info(f"Force terminating subprocess from '{self.name}': {child.pid}")
                            child.terminate()
                            try:
                                child.wait(timeout=2)
                            except psutil.TimeoutExpired:
                                child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except (ImportError, Exception) as e:
                logger.debug(f"Could not perform subprocess force cleanup: {e}")
                
    def _force_gc_cleanup(self) -> None:
        """Force garbage collection to clean up lingering references."""
        try:
            gc.collect()
            logger.debug("Garbage collection triggered successfully.")
        except Exception as e:
            logger.debug(f"Error during garbage collection: {e}")


class MCPTool(Tool):
    """
    Wrapper for an MCP tool, making it compatible with DSPy agents.
    It routes calls to the correct MCP server via the MCPServerManager.
    """

    def __init__(self, manager: 'MCPServerManager', server_name: str, tool_info: Any):
        """Create a DSPy Tool from an MCP tool description."""
        self.manager = manager
        self.server_name = server_name
        self._raw_tool_info = tool_info

        # Extract necessary information
        name, desc, input_schema = self._extract_tool_info(tool_info)
        args, arg_types, arg_desc = map_json_schema_to_tool_args(input_schema)

        # Initialize the Tool superclass
        super().__init__(
            func=self.call_tool_async,
            name=name,
            desc=desc,
            args=args,
            arg_types=arg_types,
            arg_desc=arg_desc
        )
        logger.debug(f"Created MCPTool wrapper: name='{self.name}', server='{self.server_name}'")

    def _extract_tool_info(self, tool_info: Any) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """Extract name, description and input schema from tool info."""
        name = getattr(tool_info, 'name', 'unknown_mcp_tool')
        desc = getattr(tool_info, 'description', 'No description available.')
        input_schema = getattr(tool_info, 'inputSchema', None)
        return name, desc, input_schema

    async def call_tool_async(self, **kwargs: Any) -> Any:
        """Asynchronous function called by DSPy to execute the MCP tool."""
        logger.info(f"MCPTool '{self.name}' invoked on server '{self.server_name}'")
        logger.debug(f"Arguments received: {kwargs}")

        # Get server and validate
        server = self._get_server()

        try:
            # Call the tool on the server
            result = await server.execute_tool(self.name, kwargs)
            logger.debug(f"Raw result from tool '{self.name}': {type(result)}")
            
            # Process and return the result
            return self._process_result(result)
            
        except Exception as e:
            logger.error(f"Error during execution of MCPTool '{self.name}' on server '{self.server_name}': {e}", exc_info=True)
            raise

    def _get_server(self) -> Server:
        """Get and validate the server for this tool."""
        server = self.manager.servers.get(self.server_name)
        if not server:
            raise ValueError(f"Server '{self.server_name}' required by tool '{self.name}' not found in the manager.")

        if not server._is_initialized or not server.session:
            raise RuntimeError(f"Server '{self.server_name}' is not ready for tool execution.")
        
        return server

    def _process_result(self, result: Any) -> Any:
        """Process the result from the tool execution."""
        # Handle content attribute objects
        if hasattr(result, 'content') and result.content:
            return self._process_content_result(result.content)
            
        # Handle text attribute objects
        if hasattr(result, 'text') and result.text is not None:
            logger.debug(f"Processed text attribute for tool '{self.name}'")
            return result.text
            
        # Handle dictionary results
        if isinstance(result, dict):
            return self._process_dict_result(result)
            
        # Handle primitive types
        if isinstance(result, (str, int, float, bool)):
            logger.debug(f"Processed primitive result for tool '{self.name}'")
            return result
            
        # Handle None
        if result is None:
            logger.debug(f"Tool '{self.name}' returned None.")
            return "Tool executed successfully but returned no specific content."
            
        # Default fallback for other types
        logger.debug(f"Processed result with default str() for tool '{self.name}'")
        return str(result)

    def _process_content_result(self, content: Any) -> str:
        """Process content attribute from tool result."""
        if isinstance(content, list) and content:
            try:
                text_parts = [str(getattr(item, 'text', item)) for item in content]
                processed_result = "\n".join(filter(None, text_parts))
                logger.debug(f"Processed list content for tool '{self.name}'")
                return processed_result or "Tool executed, list content processed but resulted in empty string."
            except Exception as e:
                logger.warning(f"Could not process list content for tool '{self.name}': {e}. Falling back to string representation.")
                return str(content)
        else:
            logger.debug(f"Processed non-list content for tool '{self.name}'")
            return str(content)

    def _process_dict_result(self, result: Dict[str, Any]) -> str:
        """Process dictionary result from tool execution."""
        # Check for common patterns in results
        for key in ("message", "output", "result", "text"):
            if key in result:
                return str(result[key])
                
        # Fallback: pretty print the dictionary
        logger.debug(f"Processed dictionary result for tool '{self.name}'")
        return json.dumps(result, indent=2)


class MCPServerManager:
    """
    Manages multiple MCP server connections and provides DSPy-compatible tools.
    
    Use as an async context manager.
    """

    def __init__(self):
        """Initialize the MCP server manager."""
        self.servers: Dict[str, Server] = {}
        disable_logging()  # Disable logging by default
        logger.info("MCPServerManager initialized.")

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from a JSON file."""
        MCPServerManager._validate_file_path(file_path)
        
        logger.info(f"Loading MCP server configuration from: {file_path}")
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully.")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading config: {e}")
            raise

    @staticmethod
    def _validate_file_path(file_path: str) -> None:
        """Validate the config file path."""
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Invalid file path provided.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

    async def initialize_servers(self, config: Dict[str, Any]) -> None:
        """Initialize all servers defined in the configuration dictionary."""
        self._validate_server_config(config)
        
        server_configs = config["mcpServers"]
        if not server_configs:
            logger.warning("No servers defined in the 'mcpServers' section of the configuration.")
            return

        logger.info(f"Initializing {len(server_configs)} MCP server(s)...")
        
        # Prepare servers and tasks
        init_tasks, servers_to_add = self._prepare_server_initialization(server_configs)
        
        if not init_tasks:
            logger.warning("No valid server configurations found to initialize.")
            return

        # Execute all initialization tasks
        await self._execute_initialization_tasks(init_tasks, servers_to_add)

    def _validate_server_config(self, config: Dict[str, Any]) -> None:
        """Validate the server configuration format."""
        if "mcpServers" not in config or not isinstance(config["mcpServers"], dict):
            raise ValueError("Configuration must contain an 'mcpServers' dictionary.")

    def _prepare_server_initialization(self, server_configs: Dict[str, Any]) -> Tuple[List[asyncio.Task], Dict[str, Server]]:
        """Prepare server initialization tasks."""
        init_tasks = []
        servers_to_add = {}

        for name, server_config in server_configs.items():
            if not isinstance(server_config, dict):
                logger.error(f"Invalid configuration for server '{name}'. Skipping.")
                continue
                
            if name in self.servers:
                logger.warning(f"Server '{name}' already exists. Skipping re-initialization.")
                continue

            server = Server(name, server_config)
            servers_to_add[name] = server
            init_tasks.append(asyncio.create_task(server.initialize(), name=f"init_{name}"))

        return init_tasks, servers_to_add

    async def _execute_initialization_tasks(self, init_tasks: List[asyncio.Task], servers_to_add: Dict[str, Server]) -> None:
        """Execute all initialization tasks and process results."""
        results = await asyncio.gather(*init_tasks, return_exceptions=True)

        # Process results
        successful_initializations = 0
        for i, result in enumerate(results):
            task = init_tasks[i]
            server_name = task.get_name().split("_", 1)[1]
            server_instance = servers_to_add[server_name]

            if isinstance(result, Exception):
                logger.error(f"Failed to initialize server '{server_name}': {result}")
            else:
                logger.info(f"Server '{server_name}' added to manager.")
                self.servers[server_name] = server_instance
                successful_initializations += 1

        # Summary log
        logger.info(f"Finished server initialization. {successful_initializations}/{len(init_tasks)} servers initialized successfully.")
        if successful_initializations < len(init_tasks):
            logger.warning("Some servers failed to initialize. Check logs for details.")

    async def get_all_tools(self) -> List[MCPTool]:
        """Retrieve and wrap all available tools from all initialized servers."""
        all_mcp_tools: List[MCPTool] = []
        
        if not self.servers:
            logger.warning("No servers initialized, cannot retrieve tools.")
            return all_mcp_tools

        logger.info(f"Fetching tools from {len(self.servers)} initialized server(s)...")
        
        # Prepare tool listing tasks
        list_tool_tasks, server_names_in_order = self._prepare_tool_listing_tasks()
        
        if not list_tool_tasks:
            logger.warning("No initialized servers available to list tools from.")
            return all_mcp_tools

        # Execute tool listing tasks
        results = await asyncio.gather(*list_tool_tasks, return_exceptions=True)
        
        # Process results
        all_mcp_tools = self._process_tool_listing_results(results, server_names_in_order)
        
        logger.info(f"Total MCPTools created: {len(all_mcp_tools)}")
        return all_mcp_tools

    def _prepare_tool_listing_tasks(self) -> Tuple[List[asyncio.Task], List[str]]:
        """Prepare tasks for listing tools from servers."""
        list_tool_tasks = []
        server_names_in_order = []

        for name, server in self.servers.items():
            if server._is_initialized:
                list_tool_tasks.append(asyncio.create_task(server.list_tools(), name=f"list_{name}"))
                server_names_in_order.append(name)
            else:
                logger.warning(f"Skipping tool listing for server '{name}' as it is not initialized.")

        return list_tool_tasks, server_names_in_order

    def _process_tool_listing_results(self, results: List[Any], server_names: List[str]) -> List[MCPTool]:
        """Process the results of tool listing tasks."""
        all_mcp_tools = []
        
        for i, result in enumerate(results):
            server_name = server_names[i]
            
            if isinstance(result, Exception):
                logger.error(f"Failed to list tools for server '{server_name}': {result}")
                continue
                
            if not isinstance(result, list):
                logger.warning(f"Unexpected result type ({type(result)}) when listing tools for server '{server_name}'.")
                continue
                
            logger.info(f"Received {len(result)} tools from server '{server_name}'.")
            
            # Create tool wrappers
            for tool_info in result:
                try:
                    mcp_tool_instance = MCPTool(self, server_name, tool_info)
                    all_mcp_tools.append(mcp_tool_instance)
                except Exception as e:
                    tool_name_attr = getattr(tool_info, 'name', 'unknown')
                    logger.error(f"Failed to create MCPTool wrapper for tool '{tool_name_attr}' from server '{server_name}': {e}", exc_info=True)
                    
        return all_mcp_tools

    async def cleanup(self) -> None:
        """Clean up resources for all managed server connections."""
        if not self.servers:
            logger.info("MCPServerManager cleanup: No servers to clean up.")
            return

        logger.info(f"Cleaning up {len(self.servers)} server(s)...")
        
        # Create cleanup tasks
        cleanup_tasks = [
            asyncio.create_task(server.cleanup(), name=f"cleanup_{name}")
            for name, server in self.servers.items()
        ]

        # Wait for all cleanup tasks
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = cleanup_tasks[i]
                server_name = task.get_name().split("_", 1)[1]
                logger.error(f"Error during cleanup of server '{server_name}': {result}")

        logger.info("MCPServerManager cleanup finished.")
        self.servers = {}  # Clear the servers dictionary

    async def __aenter__(self) -> 'MCPServerManager':
        """Enter the async context."""
        logger.debug("Entering MCPServerManager async context.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context and ensure cleanup."""
        logger.debug(f"Exiting MCPServerManager async context (exception: {exc_type}).")
        await self.cleanup()