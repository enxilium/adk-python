"""
WebSocketServer - Handles FastAPI setup, multi-client WebSocket connections, and message routing
"""
import json
import asyncio
import base64
import logging
import os
import uuid
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Query
import uvicorn
import httpx

from google.genai.types import Blob
from ..util.websocket_communication import set_websocket_connection, remove_websocket_connection, handle_websocket_response, get_mcp_queue
from .agent_runner import AgentRunner
from ...auth import get_user_from_token
from ...database import get_database_session, AgentUserContext
from ...database.models import User

logger = logging.getLogger(__name__)

_mcp_port_mapping = os.environ.get('MCP_PORTS', '{}')

def get_mcp_port_mapping():
    """Get the MCP port mapping that was loaded from environment variable"""
    global _mcp_port_mapping
    return json.loads(_mcp_port_mapping)

class WebSocketServer:
    """Handles multi-client WebSocket connections and message routing for dual voice/text sessions"""
    
    def __init__(self):
        self.app = FastAPI(title="Luna AI Multi-Client Streaming Server")
        
        # Multi-client connection tracking 
        self.client_websockets: Dict[str, WebSocket] = {}
        self.client_runners: Dict[str, AgentRunner] = {}
        self.client_user_contexts: Dict[str, AgentUserContext] = {}  # Track minimal user context per client
        
        # Per-client session state tracking
        self.client_voice_sessions: Dict[str, bool] = {}
        self.client_text_sessions: Dict[str, bool] = {}
        self.client_voice_tasks: Dict[str, asyncio.Task] = {}
        
        # Log MCP port mapping when server is created
        mcp_ports = get_mcp_port_mapping()
        logger.info(f"[WEBSOCKET] MCP PORTS: {mcp_ports}")
        
        # Setup routes directly in constructor
        self._setup_routes()
    
    def _generate_client_id(self) -> str:
        """Generate a unique client ID"""
        return str(uuid.uuid4())
    
    def _register_client(self, client_id: str, websocket: WebSocket, user: User):
        """Register a new client with authenticated user"""
        # Convert User to minimal AgentUserContext
        user_context = user.to_agent_context()
        
        self.client_websockets[client_id] = websocket
        self.client_runners[client_id] = AgentRunner(client_id, user_context)
        self.client_user_contexts[client_id] = user_context
        self.client_voice_sessions[client_id] = False
        self.client_text_sessions[client_id] = False
        
        # Provide websocket reference to util.py tools
        set_websocket_connection(client_id, websocket)
        
        # Pre-initialize voice agent and warm up MCP connections immediately
        asyncio.create_task(self._pre_initialize_agent(client_id))
        
        logger.info(f"Client {client_id} registered with user context: {user_context}")

    async def _pre_initialize_agent(self, client_id: str):
        """Pre-initialize agent and warm up MCP connections for instant session starts"""
        try:
            agent_runner = self.client_runners[client_id]
            await agent_runner._initialize_voice()  # This includes MCP warming
            logger.info(f"[WEBSOCKET] Pre-initialized voice agent for client {client_id} - ready for instant session start")
        except Exception as e:
            logger.error(f"[WEBSOCKET] Failed to pre-initialize agent for client {client_id}: {e}")
    
    def _setup_routes(self):
        """Set up WebSocket routes"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
            """Main WebSocket endpoint for both voice and text sessions - requires JWT authentication"""
            
            # Authenticate user before accepting WebSocket connection
            from ...database.connection import get_session_factory
            SessionLocal = get_session_factory()
            db = SessionLocal()
            
            try:
                # Validate JWT token
                user = await get_user_from_token(db, token)
                if not user:
                    await websocket.close(code=4001, reason="Invalid or expired token")
                    return
                
                # Accept WebSocket connection after successful authentication
                await websocket.accept()
                
                # Generate unique client ID
                client_id = self._generate_client_id()
                
                # Register client with user context
                self._register_client(client_id, websocket, user)
                
                logger.info(f"[WEBSOCKET] Authenticated client {client_id} connected for user {user.username}")
                
            except Exception as e:
                logger.error(f"[WEBSOCKET] Authentication error: {e}")
                await websocket.close(code=4001, reason="Authentication failed")
                return
            finally:
                db.close()
            
            try:
                logger.info(f"[WEBSOCKET] Client {client_id} ready, waiting for session initialization...")
                
                # Handle multiple session types in a loop
                while True:
                    try:
                        message_json = await websocket.receive_text()
                        message = json.loads(message_json)
                        message_type = message.get("type", "")
                        
                        # Route messages based on type
                        if message_type == "start_voice_session":
                            await self._handle_voice_session_start(client_id, websocket, message)
                        elif message_type == "start_text_session":
                            await self._handle_text_session_start(client_id, websocket, message)
                        elif message_type == "text_action":
                            await self._handle_text_action(client_id, websocket, message)
                        elif message_type == "stop_voice_session":
                            await self._handle_voice_session_stop(client_id)
                        elif message_type == "stop_text_session":
                            await self._handle_text_session_stop(client_id)
                        elif message_type in ["voice_content", "text_content", "audio", "video"]:
                            await self._route_session_message(client_id, message)
                        elif message_type.endswith("_response"):
                            handle_websocket_response(client_id, message)
                        else:
                            await self._route_session_message(client_id, message)
                            
                    except WebSocketDisconnect:
                        logger.info(f"[WEBSOCKET] Client {client_id} disconnected")
                        break
                    except json.JSONDecodeError as e:
                        logger.error(f"[WEBSOCKET] Invalid JSON received from client {client_id}: {e}")
                        continue  # Continue processing messages despite JSON errors
                    except Exception as e:
                        logger.error(f"[WEBSOCKET] Error processing message from client {client_id}: {e}")
                        continue  # Continue processing messages despite other errors
                        
            except Exception as e:
                logger.error(f"[WEBSOCKET] Fatal error for client {client_id}: {e}")
            finally:
                await self._cleanup_client(client_id)
                
        @self.app.get("/")
        async def root():
            """Root endpoint with server info"""
            return {
                "name": "Luna AI Multi-Client Streaming Server",
                "version": "1.0.0",
                "status": "running",
                "active_clients": len(self.client_websockets),
                "endpoints": {
                    "health": "/health",
                    "weather": "/weather?city={city_name}",
                    "websocket": "/ws"
                },
                "description": "Multi-client WebSocket server for Luna AI agent communication"
            }
                
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "active_clients": len(self.client_websockets),
                "active_voice_sessions": sum(self.client_voice_sessions.values()),
                "active_text_sessions": sum(self.client_text_sessions.values()),
            }

        @self.app.get("/weather")
        async def get_weather(city: str = Query(..., description="City name to get weather for")):
            """Get weather data for a specified city using OpenWeatherMap API"""
            try:
                # Get API key from environment
                api_key = os.getenv("WEATHERAPI_KEY")
                if not api_key:
                    raise HTTPException(status_code=500, detail="Weather API key not configured")
                
                # Make request to OpenWeatherMap API
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        weather_data = response.json()
                        return weather_data
                    elif response.status_code == 401:
                        raise HTTPException(status_code=500, detail="Invalid API key")
                    elif response.status_code == 404:
                        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
                    else:
                        raise HTTPException(status_code=500, detail="Weather service temporarily unavailable")
                        
            except httpx.RequestError as e:
                logger.error(f"Weather API request failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to connect to weather service")
            except Exception as e:
                logger.error(f"Weather endpoint error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.post("/mcp/{client_id}/{mcp_name}")
        async def mcp_proxy(client_id: str, mcp_name: str, request: Request):
            """Proxy MCP requests to clients"""
            if client_id not in self.client_websockets:
                raise HTTPException(status_code=404, detail="Client not connected")
            
            if mcp_name not in ["filesystem", "google"]:
                raise HTTPException(status_code=400, detail="Unknown MCP name")
            
            websocket = self.client_websockets[client_id]
            data = await request.json()
            
            # Check if this is a notification-style request (fire and forget)
            method = data.get("method", "")
            if method.startswith("notification"):
                # Fire and forget - send notification immediately without queuing
                request_id = str(uuid.uuid4())
                message = {
                    "type": "mcp_request",
                    "mcp_name": mcp_name,
                    "data": data,
                    "request_id": request_id
                }
                
                logger.info(f"[WEBSOCKET] Firing MCP notification to client {client_id} for {mcp_name} with request_id {request_id}")
                await websocket.send_text(json.dumps(message))
                
                # Return 202 Accepted for notifications (MCP client expects this for fire-and-forget)
                from fastapi.responses import Response
                return Response(status_code=202)
            
            # Handle regular request-response MCP calls
            request_id = str(uuid.uuid4())
            message = {
                "type": "mcp_request",
                "mcp_name": mcp_name,
                "data": data,
                "request_id": request_id
            }
            
            logger.info(f"[WEBSOCKET] Forwarding MCP request to client {client_id} for {mcp_name} with request_id {request_id}")
            await websocket.send_text(json.dumps(message))
            
            # Wait for response with timeout to prevent blocking
            try:
                response_data = await asyncio.wait_for(get_mcp_queue(client_id).get(), timeout=30.0)
                
                # Handle null or empty responses gracefully
                if response_data is None:
                    logger.warning(f"[WEBSOCKET] Received null response for MCP request {request_id}")
                    return {"error": "No response received from client"}
                
                result = response_data.get("data")
                
                # Handle empty data in response
                if result is None:
                    logger.warning(f"[WEBSOCKET] Received empty data in MCP response for request {request_id}")
                    return {"error": "Empty response data"}
                
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"[WEBSOCKET] Timeout waiting for MCP response for client {client_id}, request {request_id}")
                return {"error": "Request timeout - no response from client"}
            except Exception as e:
                logger.error(f"[WEBSOCKET] Error processing MCP response for client {client_id}, request {request_id}: {e}")
                return {"error": f"Internal error: {str(e)}"}

    async def _handle_voice_session_start(self, client_id: str, websocket: WebSocket, message: dict):
        """Start voice session with live streaming for a specific client"""
        if self.client_voice_sessions.get(client_id, False):
            logger.error(f"[WEBSOCKET] Voice session already active for client {client_id}")
            await websocket.send_text(json.dumps({"error": "Voice session already active"}))
            return
            
        try:
            initial_message = message.get("initial_message")
            memories = message.get("memories", [])
            
            if initial_message:
                logger.info(f"[WEBSOCKET] Starting voice session for client {client_id} with initial message: {initial_message[:50]}...")
            else:
                logger.info(f"[WEBSOCKET] Starting voice session for client {client_id} without initial message")
            
            logger.info(f"[WEBSOCKET] Voice session starting for client {client_id} with {len(memories)} memories")
            
            agent_runner = self.client_runners[client_id]
            live_events, live_request_queue = await agent_runner.start_voice_conversation(initial_message, memories=memories)
            self.client_voice_sessions[client_id] = True
            
            async def voice_message_sender(message: dict):
                try:
                    message["session"] = "voice"
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"[WEBSOCKET] Voice send error for client {client_id}: {e}")
            
            voice_events_task = asyncio.create_task(
                agent_runner.process_voice_events(live_events, voice_message_sender)
            )
            self.client_voice_tasks[client_id] = voice_events_task
            
            await websocket.send_text(json.dumps({
                "type": "voice_session_started",
                "status": "success",
                "memories_loaded": len(memories)
            }))
            
        except Exception as e:
            logger.error(f"[WEBSOCKET] Error starting voice session for client {client_id}: {e}")
            await websocket.send_text(json.dumps({
                "type": "voice_session_error",
                "error": str(e)
            }))

    async def _handle_text_session_start(self, client_id: str, websocket: WebSocket, message: dict):
        """Start text session for single request/response for a specific client"""
        try:
            action = message.get("action")
            selected_text = message.get("selected_text")
            additional_prompt = message.get("additional_prompt")
            memories = message.get("memories", [])
            
            if not action or not selected_text:
                await websocket.send_text(json.dumps({
                    "type": "text_session_error",
                    "error": "Missing action or selected_text"
                }))
                return
            
            logger.info(f"[WEBSOCKET] Processing text action for client {client_id}: {action} for text: {selected_text[:50]}...")
            
            # Initialize text session if not already active
            agent_runner = self.client_runners[client_id]
            if not self.client_text_sessions.get(client_id, False):
                await agent_runner.start_text_conversation(memories=memories)
                self.client_text_sessions[client_id] = True
                logger.info(f"[WEBSOCKET] Text session started for client {client_id} with {len(memories)} memories loaded")
            
            # Process the text action
            result = await agent_runner.process_text_action(action, selected_text, additional_prompt)
            
            # Send result back
            await websocket.send_text(json.dumps({
                "type": "text_session_result",
                "action": action,
                "result": result,
                "status": "success"
            }))
            
        except Exception as e:
            logger.error(f"[WEBSOCKET] Error processing text action for client {client_id}: {e}")
            await websocket.send_text(json.dumps({
                "type": "text_session_error",
                "error": str(e)
            }))

    async def _handle_text_action(self, client_id: str, websocket: WebSocket, message: dict):
        """Handle text overlay actions with different behaviors for a specific client"""
        try:
            action = message.get("action")
            text = message.get("text")
            additional_prompt = message.get("additional_prompt")
            memories = message.get("memories", [])
            
            if not action or not text:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": "Missing action or text"
                }))
                return
            
            logger.info(f"[WEBSOCKET] Processing text overlay action for client {client_id}: {action}")
            
            # Initialize text session if not already active (persistent for all actions)
            agent_runner = self.client_runners[client_id]
            if not self.client_text_sessions.get(client_id, False):
                await agent_runner.start_text_conversation(memories=memories)
                self.client_text_sessions[client_id] = True
                logger.info(f"[WEBSOCKET] Text session started for overlay action for client {client_id} with {len(memories)} memories")
            
            # All actions use streaming - the difference is how the frontend handles the response
            await self._stream_text_response(client_id, websocket, action, text, additional_prompt)
                
        except Exception as e:
            logger.error(f"[WEBSOCKET] Error processing text action for client {client_id}: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": str(e)
            }))

    async def _stream_text_response(self, client_id: str, websocket: WebSocket, action: str, text: str, additional_prompt: str = None):
        """Stream text response for chat/explain actions for a specific client"""
        try:
            # Use the agent runner to get a streaming async generator
            agent_runner = self.client_runners[client_id]
            async_gen = await agent_runner.stream_text_action(action, text, additional_prompt)
            
            # Stream the response in real-time
            async for event in async_gen:
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            # Send each text chunk as it arrives
                            await websocket.send_text(json.dumps({
                                "type": "chunk",
                                "data": part.text
                            }))
            
            # Send completion signal
            await websocket.send_text(json.dumps({
                "type": "complete",
                "data": ""
            }))
            
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": str(e)
            }))

    async def _handle_voice_session_stop(self, client_id: str):
        """Stop voice session for a specific client"""
        if not self.client_voice_sessions.get(client_id, False):
            return
            
        try:
            # Cancel voice events task
            if client_id in self.client_voice_tasks:
                voice_task = self.client_voice_tasks[client_id]
                if not voice_task.done():
                    voice_task.cancel()
                    try:
                        await voice_task
                    except asyncio.CancelledError:
                        pass
                del self.client_voice_tasks[client_id]
            
            # End voice conversation
            agent_runner = self.client_runners[client_id]
            await agent_runner.end_voice_conversation()
            self.client_voice_sessions[client_id] = False
            
            logger.info(f"[WEBSOCKET] Voice session stopped for client {client_id}")
            
        except Exception as e:
            logger.error(f"[WEBSOCKET] Error stopping voice session for client {client_id}: {e}")

    async def _handle_text_session_stop(self, client_id: str):
        """Stop text session for a specific client"""
        if not self.client_text_sessions.get(client_id, False):
            return
            
        try:
            agent_runner = self.client_runners[client_id]
            await agent_runner.end_text_conversation()
            self.client_text_sessions[client_id] = False
            logger.info(f"[WEBSOCKET] Text session stopped for client {client_id}")
            
        except Exception as e:
            logger.error(f"[WEBSOCKET] Error stopping text session for client {client_id}: {e}")

    async def _route_session_message(self, client_id: str, message: dict):
        """Route messages to appropriate active session for a specific client"""
        message_type = message.get("type", "")
        
        # Route voice-related messages
        if message_type in ["voice_content", "audio", "video"]:
            if not self.client_voice_sessions.get(client_id, False):
                return
                
            agent_runner = self.client_runners[client_id]
            if not hasattr(agent_runner, 'voice_live_request_queue') or not agent_runner.voice_live_request_queue:
                return
                
            mime_type = message.get("mime_type", "")
            data = message.get("data", "")
            
            if message_type == "voice_content":
                content = message.get("content", "")
                logger.info(f"[WEBSOCKET] Sending voice content for client {client_id}: {content[:50]}...")
                await agent_runner.send_voice_content(content)
                
            elif (message_type == "audio" and mime_type == "audio/pcm") or \
                 (message_type == "video" and mime_type == "image/jpeg"):
                decoded_data = base64.b64decode(data)
                agent_runner.voice_live_request_queue.send_realtime(Blob(data=decoded_data, mime_type=mime_type))
        
        # Route text-related messages 
        elif message_type == "text_content":
            # Text content is handled via start_text_session, not ongoing messaging
            logger.error(f"[WEBSOCKET] Unexpected text_content message for client {client_id} - use start_text_session instead")

    async def _cleanup_client(self, client_id: str):
        """Clean up all sessions for a specific client"""
        try:
            # Stop voice session
            await self._handle_voice_session_stop(client_id)
            
            # Stop text session
            await self._handle_text_session_stop(client_id)
            
            # Remove client references
            if client_id in self.client_websockets:
                del self.client_websockets[client_id]
            if client_id in self.client_runners:
                del self.client_runners[client_id]
            if client_id in self.client_user_contexts:
                del self.client_user_contexts[client_id]
            if client_id in self.client_voice_sessions:
                del self.client_voice_sessions[client_id]
            if client_id in self.client_text_sessions:
                del self.client_text_sessions[client_id]
            
            # Remove from websocket communication util
            remove_websocket_connection(client_id)
            
            logger.info(f"[WEBSOCKET] All sessions cleaned up for client {client_id}")
            
        except Exception as e:
            logger.error(f"[WEBSOCKET] Error during cleanup for client {client_id}: {e}")

    async def start_server(self, host: str = "0.0.0.0", port: int = None):
        """Start the FastAPI server with deployment-ready defaults"""
        if port is None:
            port = int(os.environ.get("PORT"))
            
        logger.info(f"[WEBSOCKET] Starting server on {host}:{port}")
        
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info",
            log_config=None,  # Disable uvicorn's default logging
            access_log=False,  # Disable access logging
            use_colors=False
        )
        server = uvicorn.Server(config)
        await server.serve()
