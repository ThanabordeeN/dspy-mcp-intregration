"""
Enhanced resource management for Model Context Protocol (MCP) in DSPy.

This module provides robust utilities for managing MCP server connections,
including support for multiple concurrent sessions with proper resource handling,
and integrates MCP tools seamlessly into the DSPy framework.
"""

import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple, Type
import sys
from pathlib import Path
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

    # Assuming standard installation or PYTHONPATH includes dspy root
from dspy.primitives.tool import Tool


# Configure logging with more control
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
    """Completely disable all logging from the MCP module.
    
    This function sets the logger level to CRITICAL+1 to effectively silence
    all log messages, and removes any existing handlers.
    """
    import logging
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

# Modified logger initialization to allow disabling
logger = logging.getLogger("dspy.mcp")
# Default setup - can be modified by calling setup_logging() later
setup_logging(log_level=logging.INFO)


def map_json_schema_to_tool_args(
    schema: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Type], Dict[str, str]]:
    """
    Maps a JSON schema to tool arguments compatible with DSPy Tool.

    Args:
        schema: JSON schema describing tool arguments, or None.

    Returns:
        A tuple of (args, arg_types, arg_desc) for the Tool constructor.
        Defaults to empty dicts if schema is None or lacks 'properties'.
    """
    args: Dict[str, Any] = {}
    arg_types: Dict[str, Type] = {}
    arg_desc: Dict[str, str] = {}

    if schema and "properties" in schema:
        for name, prop in schema["properties"].items():
            # DSPy Tool expects InputField/OutputField definitions or simple dicts
            # For simplicity here, we just store the property schema itself.
            # DSPy's internal mechanisms will handle this.
            args[name] = prop

            # Basic type mapping (can be expanded if needed)
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
        # Initialize exit_stack here, it will be used in initialize and cleanup
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._is_initialized: bool = False
        logger.info(f"Server instance '{self.name}' created.")


    async def initialize(self) -> None:
        """
        Initialize the server connection using an AsyncExitStack for resource management.

        Raises:
            ValueError: If command is None or invalid.
            Exception: If server initialization fails.
        """
        if self._is_initialized:
            logger.warning(f"Server '{self.name}' already initialized.")
            return
        if not self.exit_stack:
             self.exit_stack = AsyncExitStack() # Ensure stack exists if cleanup was called

        logger.info(f"Initializing server '{self.name}'...")
        command_name = self.config.get("command")
        if not command_name:
             raise ValueError(f"Missing 'command' in config for server '{self.name}'")

        # Resolve command path (e.g., finding 'npx' or other executables)
        command_path = (
            shutil.which("npx")
            if command_name == "npx"
            else shutil.which(command_name) # Try finding other commands in PATH
        )
        # If not found directly, assume it might be a relative/absolute path
        if command_path is None:
             command_path = command_name # Use the provided name as path

        if not command_path: # Still couldn't resolve
            raise ValueError(f"Command '{command_name}' not found or invalid for server '{self.name}'.")

        logger.info(f"Resolved command for server '{self.name}': {command_path}")

        server_params = StdioServerParameters(
            command=command_path,
            args=self.config.get("args", []), # Default to empty list if missing
            env={**os.environ, **self.config.get("env", {})} # Merge env vars
        )

        try:
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
            self._is_initialized = True
            logger.info(f"Server '{self.name}' initialized successfully.")

        except Exception as e:
            logger.error(f"Error initializing server '{self.name}': {e}", exc_info=True)
            # Attempt cleanup immediately if initialization fails
            await self.cleanup()
            raise  # Re-raise the exception

    async def list_tools(self) -> List[Any]:
        """
        List available tools from the server.

        Returns:
            A list of available tool objects provided by the mcp-client library.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session or not self._is_initialized:
            raise RuntimeError(f"Server '{self.name}' is not initialized or session is not available.")

        logger.debug(f"Listing tools for server '{self.name}'...")
        tools_response = await self.session.list_tools()
        logger.debug(f"Received {len(tools_response.tools)} tools from server '{self.name}'.")
        # The response object has a 'tools' attribute which is the list
        return tools_response.tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 1, # Reduced default retries, adjust if needed
        delay: float = 1.0,
    ) -> Any:
        """
        Execute a tool on the server with a retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments dictionary.
            retries: Number of retry attempts (0 means one initial try).
            delay: Delay between retries in seconds.

        Returns:
            The result of the tool execution.

        Raises:
            RuntimeError: If the server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session or not self._is_initialized:
            raise RuntimeError(f"Server '{self.name}' is not initialized or session is not available.")

        attempt = 0
        last_exception = None
        while attempt <= retries:
            try:
                logger.info(f"Executing tool '{tool_name}' on server '{self.name}' (Attempt {attempt + 1}/{retries + 1})...")
                logger.debug(f"Arguments: {arguments}")
                result = await self.session.call_tool(tool_name, arguments)
                logger.info(f"Tool '{tool_name}' executed successfully on server '{self.name}'.")
                # logger.debug(f"Raw result: {result}") # Be careful logging potentially large results
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
                    raise last_exception # Raise the last encountered exception

        # Should not be reachable if retries >= 0, but added for safety
        raise RuntimeError(f"Tool execution failed unexpectedly for '{tool_name}' after retries.")


    async def cleanup(self) -> None:
        """Clean up server resources managed by the AsyncExitStack."""
        if not self._cleanup_lock.locked(): # Prevent concurrent cleanup calls
            async with self._cleanup_lock:
                if self.exit_stack:
                    logger.info(f"Cleaning up resources for server '{self.name}'...")
                    try:
                        # First try to properly close session if available
                        if self.session and hasattr(self.session, 'shutdown') and callable(self.session.shutdown):
                            try:
                                await self.session.shutdown()
                                logger.debug(f"Session for server '{self.name}' shut down.")
                            except Exception as e:
                                logger.debug(f"Error shutting down session for server '{self.name}': {e}")
                        
                        # Use a new event loop and task to ensure context consistency
                        await self.exit_stack.aclose()
                        logger.info(f"Resources for server '{self.name}' cleaned up.")
                    except RuntimeError as e:
                        if "Attempted to exit cancel scope in a different task" in str(e):
                            # Handle the specific error gracefully
                            logger.warning(f"Task context error during cleanup of server '{self.name}'. "
                                         f"This is likely due to async task boundaries - resources may still be cleaned up by Python's GC.")
                            # Force cleanup of any potential subprocess resources
                            self._force_close_subprocesses()
                        else:
                            # Log error during cleanup but don't prevent other cleanup
                            logger.error(f"Error during cleanup of server '{self.name}': {e}")
                            self._force_close_subprocesses()
                    except Exception as e:
                        # Log error during cleanup but don't prevent other cleanup
                        logger.error(f"Error during cleanup of server '{self.name}': {e}")
                        self._force_close_subprocesses()
                    finally:
                        # Reset state after cleanup
                        self.session = None
                        self._is_initialized = False
                        # Create a new stack for potential re-initialization
                        self.exit_stack = AsyncExitStack()
                        
                        # Force garbage collection to help clean up lingering references
                        self._force_gc_cleanup()
                else:
                     logger.debug(f"Cleanup called for server '{self.name}', but no active exit stack found (already cleaned up or never initialized?).")

    def _force_close_subprocesses(self):
        """Force close any subprocesses that may have been created by this server.
        
        This is a last resort to prevent resource leaks when normal cleanup fails.
        """
        # Check if we're on Windows - if so, try to find and terminate child processes by our server name
        import platform
        if platform.system() == 'Windows':
            try:
                # Try to find processes related to this server
                import psutil
                current_process = psutil.Process()
                for child in current_process.children(recursive=True):
                    # Look for server commands in command line
                    cmd_line = " ".join(child.cmdline()).lower()
                    if self.name.lower() in cmd_line or (
                        'command' in self.config and 
                        self.config['command'].lower() in cmd_line
                    ):
                        try:
                            logger.info(f"Force terminating subprocess from '{self.name}': {child.pid}")
                            child.terminate()
                            # Wait a bit and kill if it's still running
                            try:
                                child.wait(timeout=2)
                            except psutil.TimeoutExpired:
                                child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except (ImportError, Exception) as e:
                logger.debug(f"Could not perform subprocess force cleanup: {e}")
                
    def _force_gc_cleanup(self):
        """Force garbage collection to clean up lingering references."""
        import gc
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
        """
        Create a DSPy Tool from an MCP tool description.

        Args:
            manager: The MCPServerManager instance managing the server connection.
            server_name: The name of the server hosting this tool.
            tool_info: The tool information object obtained from session.list_tools().
                       Expected to have 'name', 'description', and 'inputSchema' attributes.
        """
        self.manager = manager
        self.server_name = server_name
        self._raw_tool_info = tool_info # Store for potential future use

        # Extract necessary information safely
        name = getattr(tool_info, 'name', 'unknown_mcp_tool')
        desc = getattr(tool_info, 'description', 'No description available.')
        input_schema = getattr(tool_info, 'inputSchema', None)

        # Add server context to description for clarity if multiple servers exist
        # desc = f"[{self.server_name}] {desc}" # Optional: uncomment if helpful for LLM

        # Map JSON schema to DSPy Tool arguments
        args, arg_types, arg_desc = map_json_schema_to_tool_args(input_schema)

        # Pass the async execution method directly as the function
        super().__init__(
            func=self.call_tool_async, # Pass the coroutine function itself
            name=name,
            desc=desc,
            args=args,
            arg_types=arg_types,
            arg_desc=arg_desc
        )
        logger.debug(f"Created MCPTool wrapper: name='{self.name}', server='{self.server_name}'")

    async def call_tool_async(self, **kwargs: Any) -> Any:
        """
        Asynchronous function called by DSPy to execute the MCP tool.

        Args:
            **kwargs: Arguments provided by the DSPy agent for the tool.

        Returns:
            The processed result from the MCP tool execution.

        Raises:
            ValueError: If the associated server is not found in the manager.
            Exception: Propagates exceptions from the underlying server.execute_tool call.
        """
        logger.info(f"MCPTool '{self.name}' invoked on server '{self.server_name}'")
        logger.debug(f"Arguments received: {kwargs}")

        server = self.manager.servers.get(self.server_name)
        if not server:
            raise ValueError(f"Server '{self.server_name}' required by tool '{self.name}' not found in the manager.")

        if not server._is_initialized or not server.session:
             raise RuntimeError(f"Server '{self.server_name}' is not ready for tool execution.")

        try:
            # Call the tool on the specified server using the manager's reference
            result = await server.execute_tool(self.name, kwargs)
            logger.debug(f"Raw result from tool '{self.name}': {type(result)}")

            # --- Result Processing Logic ---
            # Try to extract meaningful text content, adapt as needed based on tool outputs
            if hasattr(result, 'content') and result.content:
                content = result.content
                if isinstance(content, list) and content:
                    # If content is a list, try joining text parts
                    try:
                        text_parts = [str(getattr(item, 'text', item)) for item in content]
                        processed_result = "\n".join(filter(None, text_parts))
                        logger.debug(f"Processed list content for tool '{self.name}'")
                        return processed_result or "Tool executed, list content processed but resulted in empty string."
                    except Exception as e:
                        logger.warning(f"Could not process list content for tool '{self.name}': {e}. Falling back to string representation.")
                        return str(content) # Fallback for lists
                else:
                    # If content is not a list, convert to string
                    logger.debug(f"Processed non-list content for tool '{self.name}'")
                    return str(content)
            elif hasattr(result, 'text') and result.text is not None:
                 logger.debug(f"Processed text attribute for tool '{self.name}'")
                 return result.text # Directly use text attribute if available
            elif isinstance(result, dict):
                # Handle common dictionary patterns
                if "message" in result: return str(result["message"])
                if "output" in result: return str(result["output"])
                if "result" in result: return str(result["result"])
                if "text" in result: return str(result["text"])
                # Fallback: return string representation of the dict
                logger.debug(f"Processed dictionary result for tool '{self.name}'")
                return json.dumps(result, indent=2) # Pretty print JSON dicts
            elif isinstance(result, (str, int, float, bool)):
                 logger.debug(f"Processed primitive result for tool '{self.name}'")
                 return result # Return primitive types directly
            elif result is None:
                 logger.debug(f"Tool '{self.name}' returned None.")
                 return "Tool executed successfully but returned no specific content."
            else:
                # Default fallback for other types
                logger.debug(f"Processed result with default str() for tool '{self.name}'")
                return str(result)

        except Exception as e:
            logger.error(f"Error during execution of MCPTool '{self.name}' on server '{self.server_name}': {e}", exc_info=True)
            # Re-raise the exception so the agent (e.g., ReAct) can handle it
            raise


class MCPServerManager:
    """
    Manages multiple MCP server connections and provides DSPy-compatible tools.

    Use as an async context manager:
    async with MCPServerManager() as manager:
        await manager.initialize_servers(config)
        tools = await manager.get_all_tools()
        # ... use tools with DSPy agent ...
    """

    def __init__(self):
        """Initialize the MCP server manager."""
        self.servers: Dict[str, Server] = {}
        # No internal exit stack needed here, each Server manages its own.
        disable_logging() # Disable logging by default, can be enabled later if needed

        logger.info("MCPServerManager initialized.")

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """
        Load server configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dictionary containing the loaded server configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            json.JSONDecodeError: If the configuration file is invalid JSON.
            ValueError: If the file path is empty or invalid.
        """
        if not file_path or not isinstance(file_path, str):
             raise ValueError("Invalid file path provided.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

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


    async def initialize_servers(self, config: Dict[str, Any]) -> None:
        """
        Initialize all servers defined in the configuration dictionary.

        Args:
            config: The server configuration dictionary, typically loaded from JSON.
                    Expected to have an "mcpServers" key containing a dictionary
                    where keys are server names and values are server configs.

        Raises:
            ValueError: If the configuration format is invalid.
            Exception: Propagates exceptions from individual server initializations.
        """
        if "mcpServers" not in config or not isinstance(config["mcpServers"], dict):
            raise ValueError("Configuration must contain an 'mcpServers' dictionary.")

        server_configs = config["mcpServers"]
        if not server_configs:
             logger.warning("No servers defined in the 'mcpServers' section of the configuration.")
             return

        logger.info(f"Initializing {len(server_configs)} MCP server(s)...")
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
            # Create a task for each server's initialization
            init_tasks.append(asyncio.create_task(server.initialize(), name=f"init_{name}"))

        if not init_tasks:
            logger.warning("No valid server configurations found to initialize.")
            return

        # Wait for all initialization tasks to complete
        results = await asyncio.gather(*init_tasks, return_exceptions=True)

        # Process results and add successfully initialized servers
        successful_initializations = 0
        for i, result in enumerate(results):
            task = init_tasks[i]
            server_name = task.get_name().split("_", 1)[1] # Extract name from task name
            server_instance = servers_to_add[server_name]

            if isinstance(result, Exception):
                logger.error(f"Failed to initialize server '{server_name}': {result}")
                # Server cleanup should have been called internally on init failure
            else:
                logger.info(f"Server '{server_name}' added to manager.")
                self.servers[server_name] = server_instance
                successful_initializations += 1

        logger.info(f"Finished server initialization. {successful_initializations}/{len(init_tasks)} servers initialized successfully.")
        if successful_initializations < len(init_tasks):
             logger.warning("Some servers failed to initialize. Check logs for details.")


    async def get_all_tools(self) -> List[MCPTool]:
        """
        Retrieve and wrap all available tools from all initialized servers.

        Returns:
            A list of MCPTool instances ready for use with DSPy agents.
        """
        all_mcp_tools: List[MCPTool] = []
        if not self.servers:
            logger.warning("No servers initialized, cannot retrieve tools.")
            return all_mcp_tools

        logger.info(f"Fetching tools from {len(self.servers)} initialized server(s)...")
        list_tool_tasks = []
        server_names_in_order = []

        # Create tasks to list tools concurrently
        for name, server in self.servers.items():
             if server._is_initialized:
                 list_tool_tasks.append(asyncio.create_task(server.list_tools(), name=f"list_{name}"))
                 server_names_in_order.append(name)
             else:
                  logger.warning(f"Skipping tool listing for server '{name}' as it is not initialized.")

        if not list_tool_tasks:
             logger.warning("No initialized servers available to list tools from.")
             return all_mcp_tools

        # Gather results from list_tools calls
        results = await asyncio.gather(*list_tool_tasks, return_exceptions=True)

        # Process results and create MCPTool instances
        for i, result in enumerate(results):
            server_name = server_names_in_order[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to list tools for server '{server_name}': {result}")
            elif isinstance(result, list):
                logger.info(f"Received {len(result)} tools from server '{server_name}'.")
                for tool_info in result:
                    try:
                        # Create the MCPTool wrapper instance
                        mcp_tool_instance = MCPTool(self, server_name, tool_info)
                        all_mcp_tools.append(mcp_tool_instance)
                    except Exception as e:
                        tool_name_attr = getattr(tool_info, 'name', 'unknown')
                        logger.error(f"Failed to create MCPTool wrapper for tool '{tool_name_attr}' from server '{server_name}': {e}", exc_info=True)
            else:
                 logger.warning(f"Unexpected result type ({type(result)}) when listing tools for server '{server_name}'.")


        logger.info(f"Total MCPTools created: {len(all_mcp_tools)}")
        return all_mcp_tools

    async def cleanup(self) -> None:
        """Clean up resources for all managed server connections."""
        if not self.servers:
            logger.info("MCPServerManager cleanup: No servers to clean up.")
            return

        logger.info(f"Cleaning up {len(self.servers)} server(s)...")
        cleanup_tasks = []
        for name, server in self.servers.items():
            # Create a task for each server's cleanup
            cleanup_tasks.append(asyncio.create_task(server.cleanup(), name=f"cleanup_{name}"))

        # Wait for all cleanup tasks to complete
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Log any errors during cleanup
        for i, result in enumerate(results):
            task = cleanup_tasks[i]
            server_name = task.get_name().split("_", 1)[1] # Extract name
            if isinstance(result, Exception):
                logger.error(f"Error during cleanup of server '{server_name}': {result}")

        logger.info("MCPServerManager cleanup finished.")
        self.servers = {} # Clear the servers dictionary

    async def __aenter__(self) -> 'MCPServerManager':
        """Enter the async context."""
        logger.debug("Entering MCPServerManager async context.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context and ensure cleanup."""
        logger.debug(f"Exiting MCPServerManager async context (exception: {exc_type}).")
        await self.cleanup()