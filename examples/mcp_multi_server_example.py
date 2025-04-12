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
import os
import dspy
# Import from the correct location within your dspy structure
from dspy.clients.mcp_resources import MCPServerManager # Assuming it's here

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


    async with MCPServerManager() as server_manager:

        config = server_manager.load_config(config_path)

        await server_manager.initialize_servers(config)

        all_mcp_tools = await server_manager.get_all_tools()

        react_agent = dspy.ReAct(
            MultiServerSignature,
            tools=all_mcp_tools, # Pass the list of MCPTool instances
            max_iters=7 # Limit the number of steps
        )


        request = "write a python script to get the current weather in New York City"

        result = await react_agent.async_forward(request=request)
        print("Final Result:", result.output)

if __name__ == "__main__":

    asyncio.run(main())
