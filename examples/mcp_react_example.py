"""
Example demonstrating how to use MCP with DSPy ReAct.

This example shows how to:
1. Set up an MCP client
2. Integrate MCP tools with DSPy ReAct
3. Run both synchronous and asynchronous calls

Requirements:
- Install DSPy: pip install dspy-ai
- Install MCP client: pip install model-context-protocol-client
"""

import asyncio
import dspy
from model_context_protocol.client import stdio_client, StdioServerParameters
from dspy.clients.mcp import create_mcp_react


# Define a simple ReAct signature for our task
class FileManipulationSignature(dspy.Signature):
    """Perform file operations using MCP tools."""
    
    request = dspy.InputField(desc="The user's request")
    output = dspy.OutputField(desc="The final response to the user")


async def main():
    # Create server parameters for stdio connection
    # Adjust this based on your MCP server setup
    server_params = StdioServerParameters(
        command="docker",  # Executable
        args=[
            "run",
            "-i",
            "--rm",
            "--mount",
            "type=bind,src=/path/to/your/workspace,dst=/projects/workspace",
            "mcp/filesystem",
            "/projects",
        ],
        env=None,  # Optional environment variables
    )

    # Set up DSPy with OpenAI (or your preferred LLM)
    dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4"))

    async with stdio_client(server_params) as (read, write):
        from model_context_protocol.client.session import ClientSession
        
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # Create a ReAct agent with MCP tools
            react_agent = await create_mcp_react(
                session, 
                FileManipulationSignature,
                max_iters=10
            )
            
            # Use the ReAct agent
            result = await react_agent.async_forward(
                request="Create a file called 'test.txt' and write 'Hello World' to it"
            )
            
            print("ReAct execution result:")
            print(f"Output: {result.output}")
            print(f"Trajectory steps: {len(result.trajectory) // 4}")  # Each step has 4 entries
            
            # If you want to inspect the full trajectory
            print("\nDetailed trajectory:")
            for i in range(len(result.trajectory) // 4):
                print(f"Step {i+1}:")
                print(f"  Thought: {result.trajectory.get(f'thought_{i}')}")
                print(f"  Tool: {result.trajectory.get(f'tool_name_{i}')}")
                print(f"  Args: {result.trajectory.get(f'tool_args_{i}')}")
                print(f"  Observation: {result.trajectory.get(f'observation_{i}')}")
                print()


if __name__ == "__main__":
    asyncio.run(main())