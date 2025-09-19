# MCP Pre-Connection Optimization

This document describes the preprocessing separation changes made to `BaseLlmFlow` to enable MCP server pre-connection and reduce initialization overhead during live connections.

## Problem

Previously, the preprocessing phase (including MCP tool initialization) was tightly coupled with the `run_live()` method. This meant that every time a live connection was established, the system had to:

1. Connect to MCP servers
2. List available tools
3. Initialize tool objects
4. Process all toolsets

This initialization overhead occurred during user interactions, potentially causing delays.

## Solution

The preprocessing has been separated into independent methods that can be called before establishing live connections.

### New Methods

#### `_prepare_llm_request(invocation_context) -> (LlmRequest, AsyncGenerator[Event, None] | None)`

Internal method that runs all preprocessing and returns both the prepared request and any events that need to be yielded.

#### `prepare_for_live_connection(invocation_context) -> LlmRequest`

Public method that fully prepares an LLM request for live connection. This includes:
- Running all request processors  
- Initializing all tools (including MCP connections)
- Setting up tool configurations

#### `warm_up_mcp_connections(invocation_context) -> None`

Targeted method that specifically pre-connects to MCP servers and caches tool listings, without running the full preprocessing pipeline.

#### Updated `run_live(invocation_context, prepared_llm_request=None)`

Now accepts an optional pre-prepared LLM request. If provided, skips preprocessing and goes directly to establishing the live connection.

## Usage Patterns

### Pattern 1: Full Pre-preparation
```python
# At application startup or agent initialization
prepared_request = await flow.prepare_for_live_connection(invocation_context)

# Later, during user interaction
async for event in flow.run_live(invocation_context, prepared_request):
    handle_event(event)
```

### Pattern 2: MCP-only Warm-up
```python
# Pre-connect just the MCP servers
await flow.warm_up_mcp_connections(invocation_context)

# Use normal flow (but MCP connections are already established)
async for event in flow.run_live(invocation_context):
    handle_event(event)
```

### Pattern 3: Hybrid Approach
```python
# Application startup: warm up MCP connections
await flow.warm_up_mcp_connections(invocation_context)

# Just before user interaction: prepare full request
prepared_request = await flow.prepare_for_live_connection(invocation_context)

# During interaction: use prepared request
async for event in flow.run_live(invocation_context, prepared_request):
    handle_event(event)
```

## Benefits

1. **Reduced Latency**: User interactions start faster as tools are already initialized
2. **Better Error Handling**: Connection issues are discovered during startup, not during user interaction
3. **Resource Efficiency**: MCP connections can be pooled and reused
4. **Backward Compatibility**: Existing code continues to work unchanged
5. **Flexible Optimization**: Choose between full preprocessing or targeted MCP warm-up

## Implementation Notes

- The `_run_one_step_async` method (used by `run_async`) has also been updated to use the new preprocessing separation
- All changes maintain backward compatibility
- Error handling ensures that preprocessing failures don't break the flow
- The separation is designed to work with the existing session management in `MCPSessionManager`

## Next Steps

To fully realize the benefits, consider implementing:

1. **Agent-level pre-warming**: Add initialization hooks to `LlmAgent`
2. **Runner-level coordination**: Pre-warm all agents in an agent tree
3. **Configuration options**: Allow users to control pre-connection behavior
4. **Caching strategies**: Implement intelligent cache invalidation for tool listings