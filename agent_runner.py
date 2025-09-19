"""
AgentRunner - Handles all agent-related operations and ADK session management
"""
import logging
from typing import Tuple, Callable, Optional

from google.adk.runners import Runner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai.types import Content, Modality, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, Part
from ..core.agent import get_agent_async, get_text_agent_async
from ..core.tools.browser_tools import cleanup_all_browser_sessions
from ...database.models import AgentUserContext

APP_NAME = "LUNA"

logger = logging.getLogger(__name__)

class AgentRunner:
    """
    Handles agent creation, session management, and ADK event processing for dual voice/text sessions.
    """
    
    def __init__(self, client_id: str, user_context: Optional[AgentUserContext] = None):
        """Initialize AgentRunner for a specific client with user context."""
        self.client_id = client_id
        self.user_context = user_context
        
        # Shared services
        self.session_service = InMemorySessionService()
        self.artifact_service = InMemoryArtifactService()
        
        # Voice agent components (existing)
        self.voice_session = None
        self.voice_agent = None
        self.voice_runner = None
        self.voice_live_request_queue = None
        self.voice_prepared_request = None  # Pre-warmed LLM request
        
        # Text agent components (new)
        self.text_session = None
        self.text_agent = None
        self.text_runner = None
        
        self.live_request_queue = LiveRequestQueue()
        # Set response modality (AUDIO for Luna) - matching old commit pattern
        modality = [Modality.AUDIO]
        self.runConfig = RunConfig(
            response_modalities=modality,
            speech_config=SpeechConfig(
                language_code="en-US", #TODO: Update to dynamic.
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name="Aoede" #TODO: Update to dynamic.
                    )
                ),
            ),
            support_cfc=True,
            streaming_mode=StreamingMode.SSE,
        )

        # Flag to track when end_conversation_session tool has been called
        self.pendingClose = False
        
        user_info = f" for user {user_context}" if user_context else ""
        logger.info(f"AgentRunner initialized for client {client_id}{user_info}")
    
    def get_user_info_for_logging(self) -> str:
        """Get user context info for logging purposes"""
        if self.user_context:
            return f"{self.user_context.username} (ID: {self.user_context.user_id}, Tier: {self.user_context.tier})"
        return "unknown user"
    
    def get_user_context(self) -> Optional[AgentUserContext]:
        """Get the user context for this agent session"""
        return self.user_context
    
    async def start_voice_conversation(self, initial_message: Optional[str] = None, memories=None) -> Tuple:
        """
        Begins a voice conversation and returns (live_events, live_request_queue)
        If initial_message is provided, it will be sent automatically after session setup
        """
        if self.voice_session is not None:
            raise RuntimeError("Voice session already active")
            
        user_info = self.get_user_info_for_logging()
        logger.info(f"[CLIENT:{self.client_id}] Starting voice conversation session for {user_info}")

        await self._initialize_voice(memories=memories)

        logger.info(f"[CLIENT:{self.client_id}] Created voice session {self.voice_session.id} for user {self.client_id}")
        
        # Use optimized run_live with pre-warmed MCP connections
        live_events = self._run_live_optimized()

        # If initial_message provided, send it immediately after session setup
        if initial_message:
            logger.info(f"[CLIENT:{self.client_id}] Sending initial message: {initial_message[:50]}...")
            self.voice_live_request_queue.send_content(
                Content(
                    role="user",
                    parts=[Part.from_text(text=initial_message)]
                )
            )

        return live_events, self.voice_live_request_queue

    async def _run_live_optimized(self):
        """Run live conversation with pre-warmed MCP connections for instant startup"""
        from google.adk.agents.invocation_context import InvocationContext
        
        # Create live invocation context
        invocation_context = InvocationContext(
            artifact_service=self.artifact_service,
            session_service=self.session_service,
            invocation_id=f"live_{self.client_id}",
            agent=self.voice_agent,
            session=self.voice_session,
            live_request_queue=self.voice_live_request_queue,
            run_config=self.runConfig,
        )
        
        # Use the pre-warmed LLM request for instant startup
        return self.voice_agent._llm_flow.run_live(invocation_context, self.voice_prepared_request)

    async def start_text_conversation(self, memories=None):
        """Start text session for simple request/response processing"""
        if self.text_session is not None:
            raise RuntimeError("Text session already active")
            
        logger.info(f"[CLIENT:{self.client_id}] Starting text conversation session")

        await self._initialize_text(memories=memories)

        logger.info(f"[CLIENT:{self.client_id}] Created text session {self.text_session.id} for user {self.client_id}")

    async def _initialize_voice(self, memories=None):
        """Initialize voice agent and session"""
        # Only initialize if not already done
        if self.voice_session is not None:
            return
            
        # Initialize voice session
        self.voice_session = await self.session_service.create_session(
            app_name=APP_NAME,
            user_id=self.client_id,
            state={
                "client_id": self.client_id
            }
        )

        self.voice_agent = await get_agent_async(self.client_id, memories=memories)

        self.voice_runner = Runner(
            app_name=APP_NAME,
            agent=self.voice_agent,
            session_service=self.session_service,
            artifact_service=self.artifact_service
        )
        
        self.voice_live_request_queue = LiveRequestQueue()
        
        # Pre-warm MCP connections for instant live startup
        await self._warm_up_mcp_connections()

    async def _warm_up_mcp_connections(self):
        """Pre-connects to all MCP servers and prepares LLM request for instant live startup"""
        from google.adk.agents.invocation_context import InvocationContext
        
        # Create a temporary invocation context for preprocessing
        temp_context = InvocationContext(
            artifact_service=self.artifact_service,
            session_service=self.session_service,
            invocation_id="temp_preprocessing",
            agent=self.voice_agent,
            session=self.voice_session,
            run_config=self.runConfig,
        )
        
        # Pre-prepare the LLM request with all MCP connections established
        self.voice_prepared_request = await self.voice_agent._llm_flow.prepare_for_live_connection(temp_context)

    async def _initialize_text(self, memories=None):
        """Initialize text agent and session"""
        # Only initialize if not already done
        if self.text_session is not None:
            return
            
        self.text_session = await self.session_service.create_session(
            app_name=APP_NAME,
            user_id=self.client_id,
            state={
                "client_id": self.client_id
            }
        )

        self.text_agent = await get_text_agent_async(self.client_id, memories=memories)

        self.text_runner = Runner(
            app_name=APP_NAME,
            agent=self.text_agent,
            session_service=self.session_service,
            artifact_service=self.artifact_service
        )

    async def process_text_action(self, action: str, selected_text: str, additional_prompt: Optional[str] = None) -> str:
        """Process single text action and return result"""
        if not self.text_session:
            raise RuntimeError("No active text session")
            
        # Build prompt based on action
        prompts = {
            "explain": f"Please explain this text clearly and concisely:\n\n{selected_text}",
            "rewrite": f"Rewrite this text to be clearer and more polished:\n\n{selected_text}",
            "chat": f"{additional_prompt or 'What can you tell me about this?'}\n\nContext (highlighted text):\n{selected_text}"
        }
        
        if action not in prompts:
            raise ValueError(f"Unsupported action: {action}")
        
        logger.info(f"[CLIENT:{self.client_id}] Processing text action '{action}' for text: {selected_text[:50]}...")
        
        # run_async returns an async generator, we need to iterate through it
        async_gen = self.text_runner.run_async(
            user_id=self.client_id,
            session_id=self.text_session.id,
            new_message=Content(
                role="user",
                parts=[Part.from_text(text=prompts[action])]
            )
        )
        
        # Collect all text parts from the async generator
        result_text = ""
        async for event in async_gen:
            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        result_text += part.text
        
        return result_text

    async def stream_text_action(self, action: str, selected_text: str, additional_prompt: Optional[str] = None):
        """Stream text action results for real-time UI updates"""
        if not self.text_session:
            raise RuntimeError("No active text session")
            
        # Build prompt based on action
        prompts = {
            "explain": f"Please explain this text clearly and concisely:\n\n{selected_text}",
            "rewrite": f"Rewrite this text to be clearer and more polished:\n\n{selected_text}",
            "chat": f"{additional_prompt or 'What can you tell me about this?'}\n\nContext (highlighted text):\n{selected_text}"
        }
        
        if action not in prompts:
            raise ValueError(f"Unsupported action: {action}")
        
        logger.info(f"[CLIENT:{self.client_id}] Streaming text action '{action}' for text: {selected_text[:50]}...")
        
        # Return the async generator directly for streaming
        return self.text_runner.run_async(
            user_id=self.client_id,
            session_id=self.text_session.id,
            new_message=Content(
                role="user",
                parts=[Part.from_text(text=prompts[action])]
            )
        )

    async def end_voice_conversation(self):
        """
        Ends the voice conversation and cleans up resources.
        """
        # Clean up browser sessions first
        try:
            await cleanup_all_browser_sessions()
        except Exception as e:
            logger.error(f"Error cleaning up browser sessions: {e}")
        
        if self.voice_session:
            await self.session_service.delete_session(
                app_name=APP_NAME,
                user_id=self.client_id,
                session_id=self.voice_session.id
            )
            self.voice_session = None
            self.voice_agent = None
            self.voice_runner = None
            self.voice_live_request_queue = None
            self.voice_prepared_request = None
            logger.info(f"[CLIENT:{self.client_id}] Voice session ended")

    async def end_text_conversation(self):
        """
        Ends the text conversation and cleans up resources.
        """
        # Clean up browser sessions first
        try:
            await cleanup_all_browser_sessions()
        except Exception as e:
            logger.error(f"Error cleaning up browser sessions: {e}")
        
        if self.text_session:
            await self.session_service.delete_session(
                app_name=APP_NAME,
                user_id=self.client_id,
                session_id=self.text_session.id
            )
            self.text_session = None
            self.text_agent = None
            self.text_runner = None
            logger.info(f"[CLIENT:{self.client_id}] Text session ended")

    async def send_voice_content(self, message: str):
        """Send content to voice agent using existing session and live_request_queue"""
        if not self.voice_session or not self.voice_live_request_queue:
            raise RuntimeError("No active voice conversation session.")
            
        self.voice_live_request_queue.send_content(
            Content(
                role="user",
                parts=[Part.from_text(text=message)]
            )
        )

    async def process_voice_events(self, live_events, message_sender: Callable) -> None:
        """
        Process voice ADK events using classify_event for all logic and send messages via callback
        """
        async for event in live_events:
            event_result = self.classify_event(event)

            match event_result["type"]:
                case "log_only":
                    logger.info(f"[CLIENT:{self.client_id}][AGENT_EVENT] {event_result['log_message']}")
                    continue

                case "audio":
                    await message_sender(event_result["websocket_message"])

                case "status":
                    logger.info(f"[CLIENT:{self.client_id}][AGENT_EVENT] {event_result['log_message']}")
                    await message_sender(event_result["websocket_message"])

                case "close_connection":
                    logger.info(f"[CLIENT:{self.client_id}][AGENT_EVENT] {event_result['log_message']}")
                    await message_sender(event_result["websocket_message"])
                    break
                    
                case _:
                    continue

    def classify_event(self, event) -> dict:
        """
        Classify event type and handle all processing logic, returning structured data for process_events
        """
        if (hasattr(event, 'turn_complete') and event.turn_complete):
            if self.pendingClose:
                return {
                    "type": "close_connection",
                    "log_message": "TURN_COMPLETE - CLOSING_CONNECTION",
                    "websocket_message": {
                        "status": "close_connection",
                    }
                }
            else:
                return {
                    "type": "status",
                    "log_message": "TURN_COMPLETE",
                    "websocket_message": {
                        "status": "turn_complete"
                    }
                }
        
        if hasattr(event, 'interrupted') and event.interrupted:
            if self.pendingClose:
                self.pendingClose = False # In case user wants to make an additional request.
            
            return {
                    "type": "status",
                    "log_message": "INTERRUPTED",
                    "websocket_message": {
                        "status": "interrupted"
                    }
                }
        
        # Handle error events (ADK uses error_code and error_message)
        if (hasattr(event, 'error_code') and event.error_code) or (hasattr(event, 'error_message') and event.error_message):
            error_code = getattr(event, 'error_code', 'unknown')
            error_message = getattr(event, 'error_message', 'no message')[:50]
            return {
                "type": "log_only",
                "log_message": f"ERROR: {error_code} - {error_message}"
            }
        
        # Handle function calls (tool requests)
        if hasattr(event, 'get_function_calls') and event.get_function_calls():
            try:
                function_calls = event.get_function_calls()
                call_names = [call.name for call in function_calls if hasattr(call, 'name')]
                
                return {
                    "type": "log_only",
                    "log_message": f"TOOL_CALL: {', '.join(call_names)}"
                }
            except:
                pass
        
        # Handle function responses (tool results)
        if hasattr(event, 'get_function_responses'):
            try:
                function_responses = event.get_function_responses()
                if function_responses:
                    response_names = [resp.name for resp in function_responses if hasattr(resp, 'name')]
                    
                    # Check for end_conversation_session tool response
                    if any(name == "end_conversation_session" for name in response_names):
                        self.pendingClose = True
                        return {
                            "type": "log_only",
                            "log_message": f"TOOL_RESULT: {', '.join(response_names)} - PENDING_CLOSE_SET"
                        }
                    
                    return {
                        "type": "log_only",
                        "log_message": f"TOOL_RESULT: {', '.join(response_names)}"
                    }
            except:
                pass
        
        # Handle code execution events (executable_code and code_execution_result)
        if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
            for part in event.content.parts:
                # Handle executable code generation
                if hasattr(part, 'executable_code') and part.executable_code:
                    code_snippet = getattr(part.executable_code, 'code', 'N/A')[:100]  # First 100 chars
                    return {
                        "type": "log_only",
                        "log_message": f"CODE_GENERATED: {code_snippet}..."
                    }
                
                # Handle code execution results
                if hasattr(part, 'code_execution_result') and part.code_execution_result:
                    outcome = getattr(part.code_execution_result, 'outcome', 'unknown')
                    output = str(getattr(part.code_execution_result, 'output', ''))  # First 50 chars
                    return {
                        "type": "log_only",
                        "log_message": f"CODE_RESULT: {outcome} - {output}..."
                    }
        
        # Handle audio content (main content type for AUDIO modality)
        if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') and event.content.parts:
            first_part = event.content.parts[0]
            if hasattr(first_part, 'inline_data') and first_part.inline_data:
                mime_type = getattr(first_part.inline_data, 'mime_type', 'unknown')
                data_size = len(getattr(first_part.inline_data, 'data', b''))
                
                # Process audio data for WebSocket transmission
                try:
                    audio_data = first_part.inline_data.data
                    import base64
                    return {
                        "type": "audio",
                        "log_message": f"AUDIO_CONTENT: {mime_type} ({data_size} bytes)",
                        "websocket_message": {
                            "type": "audio",
                            "mime_type": "audio/pcm",
                            "data": base64.b64encode(audio_data).decode("ascii")
                        }
                    }
                except (AttributeError, IndexError) as e:
                    return {
                        "type": "log_only",
                        "log_message": f"AUDIO_CONTENT_ERROR: Failed to process audio data - {e}"
                    }
        
        # Handle actions (state/artifact updates)
        if hasattr(event, 'actions') and event.actions:
            actions = []
            if hasattr(event.actions, 'state_delta') and event.actions.state_delta:
                actions.append("state_delta")
            if hasattr(event.actions, 'artifact_delta') and event.actions.artifact_delta:
                actions.append("artifact_delta")
            if hasattr(event.actions, 'transfer_to_agent') and event.actions.transfer_to_agent:
                actions.append(f"transfer_to_{event.actions.transfer_to_agent}")
            if hasattr(event.actions, 'escalate') and event.actions.escalate:
                actions.append("escalate")
            if actions:
                return {
                    "type": "log_only",
                    "log_message": f"ACTION: {', '.join(actions)}"
                }
        
        # Handle final response indicator
        if hasattr(event, 'is_final_response') and callable(event.is_final_response):
            try:
                if event.is_final_response():
                    return {
                        "type": "log_only",
                        "log_message": f"FINAL_RESPONSE"
                    }
            except:
                pass

        return {
            "type": "general",
            "log_message": f"GENERAL_EVENT: {event}",
        }