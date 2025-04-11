import asyncio
import os
from dspy.clients.mcp import MCPReactAgent
import dspy

lm = dspy.LM(
    "gemini/gemini-2.0-flash",api_key=os.getenv("GOOGLE_API_KEY")  # Will automatically check env vars if not provided
)
dspy.configure(lm=lm)

class DefaultMCPSignature(dspy.Signature):
    """Perform operations using MCP tools."""
    request = dspy.InputField(desc="The user's request")
    output = dspy.OutputField(desc="The final response to the user")

async def main():
    """
    Demonstrate the improved MCP React agent.
    """
    # Create an MCP React agent with default signature
    # You can also provide your own custom signature if needed

    agent = MCPReactAgent(DefaultMCPSignature,max_iters=10)
    
    # Set up the agent with one simple call
    # This handles server setup, LM configuration, and agent creation
    await agent.setup(
        command='npx',
        args=["-y", "@modelcontextprotocol/server-filesystem", r"F:\AI\DSPy_MCP\test"],
    )
    
    # The agent can be used as a context manager for automatic cleanup
    async with agent:
        # Run the agent with a request
        result = await agent.run("Create a file called 'test.txt' and write 'Hello World' to it")
        
        # Print the result
        print("\nReAct execution result:")
        print(f"Output: {result.output}")
        
        # Show trajectory information if available
        if hasattr(result, 'trajectory') and result.trajectory:
            print(f"\nTrajectory steps: {len(result.trajectory) // 4}")  # Each step has 4 entries
            
            # Print a detailed trajectory if requested
            print_detailed = True  # Set to False to hide detailed trajectory
            if print_detailed:
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
    

if __name__ == "__main__":
    asyncio.run(main())