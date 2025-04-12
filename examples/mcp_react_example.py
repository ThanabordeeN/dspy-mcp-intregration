import asyncio
import dspy
import os
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from dspy.clients.mcp import create_mcp_react ,cleanup_session

lm = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY")) 
dspy.configure(lm=lm)

# Define a simple ReAct signature for our task
class FileManipulationSignature(dspy.Signature):
    """Perform file operations using MCP tools."""
    request = dspy.InputField(desc="The user's request")
    output = dspy.OutputField(desc="The final response to the user")

async def main():
    server_params = StdioServerParameters(
          command='npx', # Command to run the server
          args=["-y",    # Arguments for the command
                "@modelcontextprotocol/server-filesystem",
                # IMPORTANT! This is the absolute path to the allowed directory
                r"F:\AI\DSPy_MCP\test"],
      )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # Create a ReAct agent with MCP tools
            react_agent = await create_mcp_react(
                session, 
                FileManipulationSignature,
                max_iters=10)
            # Use the ReAct agent
            result = await react_agent.async_forward(request="Create a file called 'test.txt' and write 'Hello World' to it")
            
            print("ReAct execution result:")
            print(f"Output: {result.output}")
            
            # Check if trajectory exists before trying to access its length
            if hasattr(result, 'trajectory') and result.trajectory:
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
            else:
                 print("\nNo trajectory generated or trajectory is empty.")
    await cleanup_session()

if __name__ == "__main__":
    asyncio.run(main())