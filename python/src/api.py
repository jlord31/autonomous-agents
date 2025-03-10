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
import redis
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Import your existing components
from orchestrator.supervisor_orchestrator import SupervisorOrchestrator
from utils.CreateLLMAgents import load_llm_agents
from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from tools.registry.index import get_tool_configs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #TODO: Joe - restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory cache for active orchestrators
orchestrator_cache = {}

# Try to use Redis for distributed caching if available
try:
    
    redis_client = redis.Redis(
        host=os.environ.get('REDIS_HOST', 'localhost'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        password=os.environ.get('REDIS_PASSWORD', ''),
        decode_responses=True,
        socket_timeout=5
    )
    # Test connection
    redis_client.ping()
    logger.info("Connected to Redis successfully")
    use_redis = True
except (ImportError, redis.RedisError) as e:
    logger.warning(f"Redis not available, using in-memory cache only: {str(e)}")
    use_redis = False
    redis_client = None

# Create bedrock client function
def get_bedrock_client():
    """Get or create AWS Bedrock client"""
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=os.environ.get('AWS_REGION', 'eu-west-2')
    )

# Helpers for orchestrator management
def store_orchestrator_config(user_id: str, organization_id: str, config: dict) -> str:
    """Store orchestrator config and return its ID"""
    config_id = str(uuid.uuid4())
    config_data = {
        "user_id": user_id,
        "organization_id": organization_id,
        "supervisor_model_id": config["supervisor_model_id"],
        "agent_configs": config["agent_configs"],
        "created_at": datetime.now().isoformat()
    }
    
    # Store in Redis if available
    if use_redis:
        try:
            # Store the config by ID
            redis_client.setex(
                f"orchestrator_config:{config_id}",
                86400,  # 24 hour TTL
                json.dumps(config_data)
            )
            
            # Link user to this config
            redis_client.setex(
                f"user_orchestrator:{user_id}",
                86400,  # 24 hour TTL
                config_id
            )
            
            # If organization provided, link org to this config
            if organization_id:
                redis_client.sadd(f"org_users:{organization_id}", user_id)
                redis_client.expire(f"org_users:{organization_id}", 86400)  # 24 hour TTL
                
            logger.info(f"Stored orchestrator config {config_id} in Redis")
            return config_id
        except Exception as e:
            logger.error(f"Redis error: {str(e)}")
            # Fall back to in-memory if Redis fails
    
    # In-memory fallback
    if "orchestrator_configs" not in orchestrator_cache:
        orchestrator_cache["orchestrator_configs"] = {}
    
    orchestrator_cache["orchestrator_configs"][config_id] = config_data
    
    # Link user to config
    if "user_configs" not in orchestrator_cache:
        orchestrator_cache["user_configs"] = {}
    orchestrator_cache["user_configs"][user_id] = config_id
    
    # Link org to user
    if organization_id:
        if "org_users" not in orchestrator_cache:
            orchestrator_cache["org_users"] = {}
        if organization_id not in orchestrator_cache["org_users"]:
            orchestrator_cache["org_users"][organization_id] = set()
        orchestrator_cache["org_users"][organization_id].add(user_id)
    
    logger.info(f"Stored orchestrator config {config_id} in memory")
    return config_id

def get_orchestrator_for_user(user_id: str) -> Optional[SupervisorOrchestrator]:
    """Get existing orchestrator for a user or recreate it from config"""
    # First check in-memory cache of active orchestrators
    if user_id in orchestrator_cache:
        logger.info(f"Found orchestrator for {user_id} in memory cache")
        entry = orchestrator_cache[user_id]
        entry["last_accessed"] = time.time()
        return entry["orchestrator"]
    
    # Get config ID for this user
    config_id = None
    if use_redis:
        try:
            config_id = redis_client.get(f"user_orchestrator:{user_id}")
        except Exception as e:
            logger.error(f"Redis error getting user config: {str(e)}")
    else:
        config_id = orchestrator_cache.get("user_configs", {}).get(user_id)
    
    if not config_id:
        logger.warning(f"No config found for user {user_id}")
        return None
        
    # Get the config by ID
    config = None
    if use_redis:
        try:
            config_json = redis_client.get(f"orchestrator_config:{config_id}")
            if config_json:
                config = json.loads(config_json)
        except Exception as e:
            logger.error(f"Redis error getting config: {str(e)}")
    else:
        config = orchestrator_cache.get("orchestrator_configs", {}).get(config_id)
    
    if not config:
        logger.warning(f"Config {config_id} not found")
        return None
    
    # Recreate the orchestrator from config
    try:
        bedrock_runtime = get_bedrock_client()
        
        # Create supervisor agent
        supervisor_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
            name="supervisor",
            description="The supervisor that coordinates other agents",
            model_id=config["supervisor_model_id"],
            client=bedrock_runtime
        ))
        
        # Initialize orchestrator
        orchestrator = SupervisorOrchestrator(supervisor_agent)
        
        # Load agents
        load_llm_agents(config["agent_configs"], orchestrator, bedrock_runtime)
        
        # Store in memory cache
        orchestrator_cache[user_id] = {
            "orchestrator": orchestrator,
            "last_accessed": time.time()
        }
        
        logger.info(f"Recreated orchestrator for user {user_id}")
        return orchestrator
    except Exception as e:
        logger.error(f"Error recreating orchestrator: {str(e)}")
        return None
   
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
        
        # Create supervisor
        supervisor_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
            name="supervisor",
            description="The supervisor that coordinates other agents",
            model_id=request.supervisor_model_id,
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
        
        # Store orchestrator in memory
        orchestrator_cache[request.user_id] = {
            "orchestrator": orchestrator,
            "last_accessed": time.time()
        }
        
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
        "using_redis": use_redis
    }

# Clean up inactive orchestrators periodically
@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(cleanup_inactive_orchestrators())

async def cleanup_inactive_orchestrators():
    """Remove orchestrators that haven't been used for a while"""
    while True:
        try:
            now = time.time()
            cutoff = now - 3600  # 1 hour of inactivity
            
            # Find inactive orchestrators
            to_remove = []
            for user_id, entry in orchestrator_cache.items():
                if isinstance(entry, dict) and entry.get("last_accessed", 0) < cutoff:
                    to_remove.append(user_id)
            
            # Remove them
            for user_id in to_remove:
                if user_id in orchestrator_cache:
                    del orchestrator_cache[user_id]
                    logger.info(f"Removed inactive orchestrator for user {user_id}")
            
            # Sleep for 15 minutes before checking again
            await asyncio.sleep(900)
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
            await asyncio.sleep(60)  # Retry after 1 minute if there's an error