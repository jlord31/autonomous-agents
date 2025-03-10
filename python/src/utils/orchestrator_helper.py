import uuid
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
from utils.get_bedrock_client import get_bedrock_client
import redis
import logging
import os

from orchestrator.supervisor_orchestrator import SupervisorOrchestrator
from utils.CreateLLMAgents import load_llm_agents
from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from utils.redis_client import redis_client, use_redis
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Single orchestrator cache - use this consistently
orchestrator_cache = {}

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
        
        # Store in memory cache - use the consistent variable
        orchestrator_cache[user_id] = {
            "orchestrator": orchestrator,
            "last_accessed": time.time()
        }
        
        logger.info(f"Recreated orchestrator for user {user_id}")
        return orchestrator
    except Exception as e:
        logger.error(f"Error recreating orchestrator: {str(e)}")
        return None

def store_orchestrator(user_id: str, orchestrator: SupervisorOrchestrator) -> None:
    """Store an orchestrator instance in the cache"""
    # Use the consistent cache variable
    orchestrator_cache[user_id] = {
        "orchestrator": orchestrator,
        "last_accessed": time.time()
    }
    logger.info(f"Stored orchestrator for user {user_id} in memory cache")

def update_last_accessed(user_id: str) -> None:
    """Update the 'last_accessed' timestamp for a cached orchestrator"""
    if user_id in orchestrator_cache:
        orchestrator_cache[user_id]["last_accessed"] = time.time()

def cleanup_inactive_orchestrators(timeout_seconds: int = 3600) -> List[str]:
    """Remove orchestrators that haven't been used recently
    
    Args:
        timeout_seconds: Time in seconds of inactivity before removal
        
    Returns:
        List of user_ids that were removed
    """
    now = time.time()
    cutoff = now - timeout_seconds
    
    # Find inactive orchestrators
    to_remove = []
    for user_id, entry in orchestrator_cache.items():
        # Skip special keys used for configs
        if user_id in ["orchestrator_configs", "user_configs", "org_users"]:
            continue
            
        if isinstance(entry, dict) and entry.get("last_accessed", 0) < cutoff:
            to_remove.append(user_id)
    
    # Remove them
    for user_id in to_remove:
        if user_id in orchestrator_cache:
            del orchestrator_cache[user_id]
            logger.info(f"Removed inactive orchestrator for user {user_id}")
    
    return to_remove