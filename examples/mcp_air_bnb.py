import asyncio
import os
from dspy.clients.mcp import MCPReactAgent
import dspy

lm = dspy.LM("gemini/gemini-2.0-flash",api_key=os.getenv("GOOGLE_API_KEY") )
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
    agent = MCPReactAgent(DefaultMCPSignature,max_iters=10)
    
    # This handles server setup, LM configuration, and agent creation
    await agent.setup(
                command="npx",  # Executable
        args=[
            "-y",
            "@openbnb/mcp-server-airbnb",
            "--ignore-robots-txt",
        ],  # Optional command line arguments
    )
    
    # The agent can be used as a context manager for automatic cleanup
    async with agent:
        # Run the agent with a request
        result = await agent.run("Check hotel in bangkok for 2 people from 2025-10-01 to 2025-10-05")
        
        # Print the result
        print("\nReAct execution result:")
        print(f"Output: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())