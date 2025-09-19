#!/usr/bin/env python3
"""
Example demonstrating MCP pre-connection optimization.

This example shows how to use the new preprocessing separation to pre-connect
to MCP servers before establishing live connections, reducing initialization
overhead during user interactions.
"""

import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters


class OptimizedLlmFlow(BaseLlmFlow):
    """Example flow with MCP pre-connection capabilities."""
    
    def __init__(self):
        super().__init__()
        self._prepared_requests = {}  # Cache for prepared requests

    async def run_live_optimized(
        self,
        invocation_context: InvocationContext,
        warm_up_mcp: bool = True,
    ):
        """Run live with optional MCP warm-up."""
        
        # Option 1: Pre-warm MCP connections only
        if warm_up_mcp:
            print("Warming up MCP connections...")
            await self.warm_up_mcp_connections(invocation_context)
            print("MCP connections warmed up!")
        
        # Option 2: Full preprocessing preparation
        print("Preparing LLM request...")
        prepared_request = await self.prepare_for_live_connection(invocation_context)
        print("LLM request prepared!")
        
        # Option 3: Use the prepared request for live connection
        print("Starting live connection with prepared request...")
        async for event in self.run_live(invocation_context, prepared_request):
            yield event


async def main():
    """Demonstrate the optimization approaches."""
    
    # Create an agent with MCP toolset
    mcp_toolset = MCPToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command='npx',
                args=['-y', '@modelcontextprotocol/server-filesystem', '/tmp'],
            ),
            timeout=5,
        )
    )
    
    agent = LlmAgent(
        model='gemini-2.0-flash',
        name='mcp_agent',
        instruction='Help with filesystem operations',
        tools=[mcp_toolset],
    )
    
    # Create flow and context
    flow = OptimizedLlmFlow()
    # Note: In real usage, you'd create a proper InvocationContext
    # This is just for demonstration
    
    print("=== Approach 1: MCP-only warm-up ===")
    # This would warm up just the MCP connections
    # await flow.warm_up_mcp_connections(invocation_context)
    
    print("\n=== Approach 2: Full preprocessing preparation ===") 
    # This would prepare the entire LLM request
    # prepared_request = await flow.prepare_for_live_connection(invocation_context)
    
    print("\n=== Approach 3: Use prepared request for live connection ===")
    # This would use the pre-prepared request
    # async for event in flow.run_live(invocation_context, prepared_request):
    #     print(f"Event: {event}")
    
    print("\nOptimization complete! MCP servers can now be pre-connected.")


if __name__ == "__main__":
    asyncio.run(main())