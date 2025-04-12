"""
Example demonstrating the improved MCP integration with multiple server support in DSPy.

This example shows how to:
1. Load MCP server configurations from a JSON file.
2. Initialize multiple MCP servers using the MCPServerManager.
3. Get DSPy-compatible MCPTool instances from the manager.
4. Use these tools with a DSPy agent (ReAct).
5. Handle proper resource cleanup using the async context manager.
"""

import asyncio
import logging
import os
import sys
import json # Needed for creating dummy config

# --- Path Setup ---
# Add the directory containing the 'dspy' package to sys.path
# Adjust this based on your project structure
script_dir = os.path.dirname(os.path.abspath(__file__))
# Example: If mcp_resources.py is in dspy/clients/ and this script is in examples/
project_root = os.path.abspath(os.path.join(script_dir, '..')) # Adjust '..' as needed
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

import dspy
# Import from the correct location within your dspy structure
from dspy.clients.mcp_resources import MCPServerManager # Assuming it's here

# Configure logging for the example script
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
client_logger = logging.getLogger("mcp_client_example")


# Define a simple DSPy Signature
class MultiServerSignature(dspy.Signature):
    """Perform operations using tools potentially available across multiple MCP servers."""
    request: str = dspy.InputField(desc="The user's request, potentially requiring external tools.")
    output: str = dspy.OutputField(desc="The final response to the user's request after potentially using tools.")


# --- Main Execution ---
async def main() -> None:
    """Initialize MCP servers, get tools, and run a DSPy agent."""
    lm = dspy.LM("gemini/gemini-2.0-flash",api_key=os.getenv("GOOGLE_API_KEY") )


    dspy.configure(lm=lm)
  
    # --- MCP Server Setup ---
    config_path = r"F:\AI\DSPy_MCP\dspy_dev\examples\servers_config.json" # Assumes config is in the same directory


    # Use the manager as an async context manager for automatic cleanup
    async with MCPServerManager() as server_manager:
        try:
            config = server_manager.load_config(config_path)

            client_logger.info("Initializing MCP servers...")
            await server_manager.initialize_servers(config)

            client_logger.info("Retrieving tools from initialized servers...")
            all_mcp_tools = await server_manager.get_all_tools()

            if not all_mcp_tools:
                client_logger.warning("No MCP tools were found or initialized. The agent will rely solely on the LM.")
            else:
                client_logger.info(f"Successfully retrieved {len(all_mcp_tools)} MCP tools:")
                for tool in all_mcp_tools:
                    client_logger.info(f"  - {tool.name} (from server: {tool.server_name})")


            react_agent = dspy.ReAct(
                MultiServerSignature,
                tools=all_mcp_tools, # Pass the list of MCPTool instances
                max_iters=7 # Limit the number of steps
            )
            client_logger.info("ReAct agent created with MCP tools.")


            request = "Create file 'example.txt' with the content 'Hello, world!'"

            client_logger.info(f"\n--- Sending request to ReAct agent ---")
            client_logger.info(f"Request: {request}")

            # Use async_forward since our tools are async
            result = await react_agent.async_forward(request=request)

            client_logger.info(f"\n--- Agent Response ---")
            client_logger.info(f"Final Output: {result.output}")


        except FileNotFoundError as e:
             client_logger.error(f"Configuration file error: {e}")
        except ValueError as e:
             client_logger.error(f"Configuration or initialization error: {e}")
        except RuntimeError as e:
             client_logger.error(f"Server runtime error: {e}")
        except Exception as e:
            # Catch any other unexpected errors during the process
            client_logger.error(f"An unexpected error occurred: {e}", exc_info=True)

        # Cleanup is automatically handled by the 'async with' statement exiting
        client_logger.info("MCPServerManager context exited, resources cleaned up.")


if __name__ == "__main__":
    # Ensure you have an event loop running if using certain environments (like Jupyter)
    # asyncio.run(main()) should work in standard Python scripts.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        client_logger.info("Execution interrupted by user.")