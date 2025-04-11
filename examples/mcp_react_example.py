import asyncio
import dspy
import os
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from dspy.clients.mcp import create_mcp_react ,cleanup_session


# Define a simple ReAct signature for our task
class FileManipulationSignature(dspy.Signature):
    """Perform file operations using MCP tools."""
    
    request = dspy.InputField(desc="The user's request")
    output = dspy.OutputField(desc="The final response to the user")


async def main():
    # Create server parameters for stdio connection
    # Adjust this based on your MCP server setup
    server_params = StdioServerParameters(
          command='npx', # Command to run the server
          args=["-y",    # Arguments for the command
                "@modelcontextprotocol/server-filesystem",
                # IMPORTANT! This is the absolute path to the allowed directory
                r"F:\AI\DSPy_MCP\test"],
      )

    # Set up DSPy with your preferred LLM
    # Make sure GOOGLE_API_KEY is set in your environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    lm = dspy.LM("gemini/gemini-2.0-flash", api_key=api_key) 
    dspy.configure(lm=lm)

    async with stdio_client(server_params) as (read, write):
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
    # Ensure you load environment variables if using a .env file
    # from dotenv import load_dotenv
    # load_dotenv() 
    asyncio.run(main())