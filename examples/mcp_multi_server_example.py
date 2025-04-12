"""
Example demonstrating the improved MCP integration with multiple server support.

This example shows how to:
1. Load MCP server configurations from a JSON file
2. Initialize multiple MCP servers
3. Use tools from different servers in a unified way
4. Handle proper resource cleanup
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

# Add the parent directory to sys.path to import dspy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from dspy.clients.mcp_resources import  MCPTool ,MCPServerManager


class MultiServerSignature(dspy.Signature):
    """Perform operations using tools from multiple MCP servers."""
    
    request: str = dspy.InputField(desc="The user's request")
    output: str = dspy.OutputField(desc="Response to the user's request")



async def main() -> None:
    """Initialize and run a demo with multiple MCP servers."""
    # Set up the language model
    api_key = os.getenv("GOOGLE_API_KEY")  # Default variable name for Google API key
    if not api_key:
        logging.error("Please set the OPENAI_API_KEY environment variable")
        return
        
    lm = dspy.LM(model="gemini/gemini-2.0-flash", api_key=api_key)
    dspy.configure(lm=lm)
    
    # Create a sample configuration file if it doesn't exist
    config_path = "servers_config.json"
    
    # Initialize the MCP server manager
    server_manager = MCPServerManager()
    
    try:
        # Load configuration
        config = server_manager.load_config(config_path)
        
        # Initialize servers
        logging.info("Initializing MCP servers...")
        await server_manager.initialize_servers(config)
        
        # Get all tools from all servers
        all_tools = []
        tool_map = {}  # Maps tool name to server name
        
        for server_name, server in server_manager.servers.items():
            logging.info(f"Getting tools from server: {server_name}")
            tools = await server.list_tools()
            
            for tool_info in tools:
                tool = MCPTool(server_manager, server_name, tool_info)
                all_tools.append(tool)
                
                # Store tool-to-server mapping
                if hasattr(tool_info, 'name'):
                    tool_map[tool_info.name] = server_name
                elif isinstance(tool_info, dict) and "name" in tool_info:
                    tool_map[tool_info["name"]] = server_name
        
        logging.info(f"Found {len(all_tools)} tools across {len(server_manager.servers)} servers")
        
        # Create a ReAct agent with all tools
        react_agent = dspy.ReAct(MultiServerSignature, all_tools, max_iters=10)
        
        # Demo usage with a sample request
        request = "Could you take a screenshot of the DSPy GitHub page and save it to a file called github_screenshot.png?"
        logging.info(f"\nProcessing request: {request}")
        
        result = await react_agent.async_forward(request=request)
        logging.info(f"\nResult: {result.output}")
        
        # Show trajectory for debugging
        if hasattr(result, 'trajectory'):
            steps = len(result.trajectory) // 4
            logging.info(f"\nTrajectory steps: {steps}")
            
            for i in range(steps):
                print(f"\nStep {i+1}:")
                print(f"  Thought: {result.trajectory.get(f'thought_{i}')}")
                print(f"  Tool: {result.trajectory.get(f'tool_name_{i}')}")
                print(f"  Args: {result.trajectory.get(f'tool_args_{i}')}")
                print(f"  Observation: {result.trajectory.get(f'observation_{i}')}")
        
    finally:
        # Ensure proper cleanup of all resources
        logging.info("Cleaning up server resources...")
        await server_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())