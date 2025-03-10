from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
import os
import json
import time
import boto3
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from utils.get_bedrock_client import get_bedrock_client
#from utils.redis_client import redis_client, use_redis
from utils.LRUClient import cache_store

# Import orchestratorHelper functions 
from utils.orchestrator_helper import (
    store_orchestrator_config, 
    get_orchestrator_for_user,
    store_orchestrator,
    cleanup_inactive_orchestrators
)

from orchestrator.supervisor_orchestrator import SupervisorOrchestrator
from utils.CreateLLMAgents import load_llm_agents
from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from tools.registry.index import get_tool_configs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = os.environ.get('ROOT_PATH', '')

# Data models
class SetupRequest(BaseModel):
    user_id: str
    organization_id: Optional[str] = None
    supervisor_model_id: str
    agent_configs: List[Dict]

class ChatRequest(BaseModel):
    message: str
    user_id: str
    organization_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    source: str = Field(..., description="The agent that generated the response")
    session_id: str
    metadata: Optional[Dict] = None
    
# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Orchestration API",
    description="API for managing multi-agent conversations with specialist agents",
    version="1.0.0",
    root_path=base_path,  
    openapi_url=f"{base_path}/openapi.json",  # For Swagger docs
    docs_url=f"{base_path}/docs"  # For Swagger UI
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #TODO: Joe - restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# use for streaming 
@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    await websocket.accept()
    try:
        while True:
            # Receive message from WebSocket
            data = await websocket.receive_json()
            message = data.get("message")
            
            # Get orchestrator for this user
            orchestrator = get_orchestrator_for_user(user_id)
            if not orchestrator:
                await websocket.send_json({
                    "error": "No orchestrator found for this user"
                })
                continue
                
            # Process message
            response = await orchestrator.route_request(message, user_id, session_id)
            
            # Send response
            await websocket.send_json({
                "response": response.output,
                "source": response.metadata.get("source", "unknown"),
                "metadata": response.metadata
            })
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user {user_id}")

@app.post("/api/setup", response_model=Dict)
async def setup_orchestrator(request: SetupRequest):
    """Set up a new orchestrator for a user/organization"""
    try:
        # Create bedrock client
        bedrock_runtime = get_bedrock_client()
        
        # Use user-provided supervisor model or fall back to env variable
        supervisor_model_id = request.supervisor_model_id or os.environ.get('SUPERVISOR_MODEL_ID')
        if not supervisor_model_id:
            raise HTTPException(
                status_code=400, 
                detail="No supervisor model ID provided and no default configured"
            )
        
        # Create supervisor
        supervisor_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
            name="supervisor",
            description="The supervisor that coordinates other agents",
            model_id=supervisor_model_id,
            client=bedrock_runtime
        ))
        
        # Initialize orchestrator
        orchestrator = SupervisorOrchestrator(supervisor_agent)
        
        # Load user-provided agent configurations
        load_llm_agents(request.agent_configs, orchestrator, bedrock_runtime)
        
        # Store config for persistence
        config_id = store_orchestrator_config(
            request.user_id, 
            request.organization_id or "", 
            {
                "supervisor_model_id": request.supervisor_model_id,
                "agent_configs": request.agent_configs
            }
        )
        
        # Store orchestrator in memory using helper function
        store_orchestrator(request.user_id, orchestrator)
        
        return {"status": "success", "config_id": config_id}
    except Exception as e:
        logger.error(f"Setup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a user chat message through the orchestrator"""
    try:
        # Get orchestrator for this user
        orchestrator = get_orchestrator_for_user(request.user_id)
        
        if not orchestrator:
            raise HTTPException(
                status_code=404, 
                detail="No orchestrator found for this user. Please set up first."
            )
        
        # Use provided session ID or create a new one
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        
        # Process the request through the orchestrator
        response = await orchestrator.route_request(
            request.message,
            request.user_id,
            session_id
        )
        
        # Return the formatted response
        return {
            "response": response.output,
            "source": response.metadata.get("source", "unknown"),
            "session_id": session_id,
            "metadata": response.metadata
        }
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "using_redis": len(cache_store.cache)
    }

# Clean up inactive orchestrators periodically
@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(cleanup_inactive_orchestrators_task())

# Use the helper function for cleanup
async def cleanup_inactive_orchestrators_task():
    """Background task to remove inactive orchestrators"""
    while True:
        try:
            # Call the helper function instead of directly manipulating the cache
            removed = cleanup_inactive_orchestrators(3600)  # 1 hour timeout
            if removed:
                logger.info(f"Removed {len(removed)} inactive orchestrators")
            
            # Sleep for 15 minutes before checking again
            await asyncio.sleep(900)
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
            await asyncio.sleep(60)  # Retry after 1 minute if there's an error