import asyncio
import dspy
import os
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from dspy.clients.mcp import create_mcp_react ,cleanup_session

lm = dspy.LM("gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY")) 
dspy.configure(lm=lm)

# Define a simple ReAct signature for our task
class AirbnbManipulation(dspy.Signature):
    """Perform file operations using MCP tools."""
    request = dspy.InputField(desc="The user's request")
    output = dspy.OutputField(desc="The final response to the user")

async def main():
    server_params = StdioServerParameters(
        command="npx",  # Executable
        args=[
            "-y",
            "@openbnb/mcp-server-airbnb",
            "--ignore-robots-txt",
        ],  # Optional command line arguments
        env=None,  # Optional environment variables
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # Create a ReAct agent with MCP tools
            react_agent = await create_mcp_react(
                session, 
                AirbnbManipulation,
                max_iters=10)
            # Use the ReAct agent
            result = await react_agent.async_forward(request="Check hotel in bangkok for 2 people from 2025-10-01 to 2025-10-05")
            
            print("ReAct execution result:")
            print(f"Output: {result.output}")
           
    await cleanup_session()

if __name__ == "__main__":
    asyncio.run(main())