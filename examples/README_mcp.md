# MCP Integration with DSPy ReAct

This document explains how to use the Model Context Protocol (MCP) with DSPy's ReAct framework.

## Overview

The integration enables you to use tools provided by MCP servers within DSPy's ReAct framework. The key features include:

1. **Async Support**: ReAct now has an `async_forward` method that can handle async tools
2. **Automatic Tool Integration**: Tools from an MCP server are automatically converted into DSPy Tool objects
3. **Seamless Integration**: Works with both synchronous and asynchronous contexts

## How It Works

The integration consists of three main components:

1. **Modified ReAct Class**: Adds async support to the ReAct class
2. **MCP Client Integration**: Provides utilities for working with MCP sessions
3. **MCP Tool Wrapper**: Wraps MCP tools so they can be used with DSPy

## Usage

### 1. Import Required Modules

```python
import asyncio
import dspy
from model_context_protocol.client import stdio_client, StdioServerParameters
from dspy.clients.mcp import create_mcp_react
```

### 2. Define a ReAct Signature

```python
class FileManipulationSignature(dspy.Signature):
    """Perform file operations using MCP tools."""
    
    request = dspy.InputField(desc="The user's request")
    output = dspy.OutputField(desc="The final response to the user")
```

### 3. Set Up MCP and Create a ReAct Agent

```python
# Set up MCP server parameters
server_params = StdioServerParameters(
    command="docker",
    args=[
        "run",
        "-i",
        "--rm",
        "--mount",
        "type=bind,src=/path/to/workspace,dst=/projects/workspace",
        "mcp/filesystem",
        "/projects",
    ],
)

# Connect to MCP server
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
```

### 4. Use the ReAct Agent

```python
# Use the ReAct agent asynchronously
result = await react_agent.async_forward(
    request="Create a file called 'test.txt' and write 'Hello World' to it"
)

# Print the result
print(f"Output: {result.output}")
```

## Advanced Usage

### Creating Individual MCP Tools

If you want to create individual tools from MCP:

```python
from dspy.clients.mcp import MCPClient, MCPTool

async def get_individual_tools(session):
    client = MCPClient(session)
    tools = await client.get_dspy_tools()
    return tools

# Then you can use these tools directly or create a ReAct agent with a subset of tools
tool_list = await get_individual_tools(session)
custom_react = dspy.ReAct(FileManipulationSignature, tools=tool_list[:3], max_iters=10)
```

### Using MCP Tools in Synchronous Code

The integration handles both synchronous and asynchronous contexts. If you call a function that returns a coroutine from a synchronous context, it will automatically run the coroutine in an event loop:

```python
# This will work even in a synchronous context
result = react_agent(request="List all files in the directory")
```

## Common Issues and Solutions

### Issue: "RuntimeError: Event loop is already running"

**Solution**: If you're in a Jupyter notebook or another environment where an event loop is already running, use:

```python
import nest_asyncio
nest_asyncio.apply()
```

### Issue: Tools not working as expected

**Solution**: Make sure the MCP server is properly configured and accessible. You can test individual tools before using them with ReAct:

```python
# Test a tool directly
tool_result = await session.call_tool("list_directory", arguments={"path": "/projects"})
print(tool_result)
```

## Complete Example

See the full example in [`mcp_react_example.py`](mcp_react_example.py) for a working demonstration.