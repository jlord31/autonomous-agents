from typing import Dict, List
from multi_agent_orchestrator.agents import ( BedrockLLMAgent, BedrockLLMAgentOptions)

from orchestrator.supervisor_orchestrator import SupervisorOrchestrator

def load_llm_agents(agent_configs: List[Dict], orchestrator: SupervisorOrchestrator, bedrock_runtime) -> None:
    """
    Load and create agents from a configuration array
    
    Args:
        agent_configs: List of agent configuration dictionaries
        orchestrator: The SupervisorOrchestrator to add agents to
        bedrock_runtime: The Bedrock runtime client
    """
    for agent_config in agent_configs:
        # Extract required parameters
        name = agent_config["name"]
        description = agent_config["description"]
        model_id = agent_config["model_id"]
        
        # Extract optional parameters with defaults
        streaming = agent_config.get("streaming", False)
        
        # Create options object
        agent_options = BedrockLLMAgentOptions(
            name=name,
            description=description,
            model_id=model_id,
            client=bedrock_runtime,
            streaming=streaming
        )
        
        # Handle callbacks if provided
        if "callback_class" in agent_config:
            callback_class_name = agent_config["callback_class"]
            # Dynamically import and instantiate callback class if needed
            # This is a simple version - you might need to adjust based on your actual callback classes
            if callback_class_name == "BedrockLLMAgentCallbacks":
                from utils.BedrockLLMAgentCallbacks import BedrockLLMAgentCallbacks
                agent_options.callbacks = BedrockLLMAgentCallbacks()
        
        # Handle tools if provided
        if "tools" in agent_config and agent_config["tools"]:
            agent_options.tools = agent_config["tools"]
        
        # Create the agent
        agent = BedrockLLMAgent(agent_options)
        
        # Add to orchestrator
        orchestrator.add_agent(agent)
        
    print(f"Loaded {len(agent_configs)} agents into orchestrator")
    print(f"Available agents: {orchestrator.list_agents()}")
    
# Example usage
# Example agent configurations
# TODO: Add type llm to distinguish from other agent types
""" agent_configs = [
    {
        "name": "tech_agent",
        "description": "Specialized in technology areas including software development, hardware, AI, cyber security",
        "model_id": MODEL_ID,
        "streaming": True,
        "callback_class": "BedrockLLMAgentCallbacks"
    },
    {
        "name": "travel_agent",
        "description": "Helps users book and manage flight reservations, hotels, and provides travel recommendations",
        "model_id": MODEL_ID,
        "streaming": True,
        "callback_class": "BedrockLLMAgentCallbacks"
    },
    {
        "name": "finance_agent",
        "description": "Provides financial advice, investment strategies, budget planning",
        "model_id": MODEL_ID,
        "streaming": True,
        "callback_class": "BedrockLLMAgentCallbacks"
    }
]

# Create the orchestrator with supervisor
orchestrator = SupervisorOrchestrator(supervisor_agent)

# Load agents from config
load_agents_from_config(agent_configs, orchestrator, bedrock_runtime) """