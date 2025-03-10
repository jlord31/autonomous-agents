import uuid
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import logging
import os

from orchestrator.supervisor_orchestrator import SupervisorOrchestrator
from utils.CreateLLMAgents import load_llm_agents
from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from utils.get_bedrock_client import get_bedrock_client
from utils.LRUClient import cache_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory cache specifically for active orchestrator instances
# This is separate from the LRU cache because orchestrator instances are complex objects
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
    
    # Store configuration in LRU cache
    cache_store.set(f"orchestrator_config:{config_id}", json.dumps(config_data), ttl=86400)  # 24 hour TTL
    
    # Link user to this config
    cache_store.set(f"user_orchestrator:{user_id}", config_id, ttl=86400)
    
    # If organization provided, link org to this user
    if organization_id:
        # Get existing org users or create new list
        org_users_json = cache_store.get(f"org_users:{organization_id}")
        org_users = json.loads(org_users_json) if org_users_json else []
        
        # Add user if not already in list
        if user_id not in org_users:
            org_users.append(user_id)
            cache_store.set(f"org_users:{organization_id}", json.dumps(org_users), ttl=86400)
    
    logger.info(f"Stored orchestrator config {config_id} for user {user_id}")
    return config_id

def get_orchestrator_for_user(user_id: str) -> Optional[SupervisorOrchestrator]:
    """Get existing orchestrator for a user or recreate it from config"""
    # First check in-memory cache of active orchestrators
    if user_id in orchestrator_cache:
        logger.info(f"Found orchestrator for {user_id} in memory cache")
        entry = orchestrator_cache[user_id]
        entry["last_accessed"] = time.time()
        return entry["orchestrator"]
    
    # Get config ID from LRU cache
    config_id = cache_store.get(f"user_orchestrator:{user_id}")
    
    if not config_id:
        logger.warning(f"No config found for user {user_id}")
        return None
        
    # Get the config by ID from LRU cache
    config_json = cache_store.get(f"orchestrator_config:{config_id}")
    
    if not config_json:
        logger.warning(f"Config {config_id} not found")
        return None
    
    try:
        # Parse the config JSON
        config = json.loads(config_json)
        
        # Recreate the orchestrator from config
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
    """Store an orchestrator instance in the in-memory cache"""
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
        # Skip any non-orchestrator entries
        if not isinstance(entry, dict) or "orchestrator" not in entry:
            continue
            
        if entry.get("last_accessed", 0) < cutoff:
            to_remove.append(user_id)
    
    # Remove them
    for user_id in to_remove:
        if user_id in orchestrator_cache:
            del orchestrator_cache[user_id]
            logger.info(f"Removed inactive orchestrator for user {user_id}")
    
    return to_remove

def get_users_for_organization(organization_id: str) -> List[str]:
    """Get all users associated with an organization"""
    org_users_json = cache_store.get(f"org_users:{organization_id}")
    if org_users_json:
        return json.loads(org_users_json)
    return []

def get_all_active_user_ids() -> List[str]:
    """Get IDs of all users with active orchestrators"""
    return [
        user_id for user_id in orchestrator_cache.keys()
        if isinstance(orchestrator_cache[user_id], dict) and "orchestrator" in orchestrator_cache[user_id]
    ]