import uuid
import os
import asyncio
import boto3
from typing import Dict, List
import dotenv
from dotenv import load_dotenv

# Import our custom components
from SupervisorOrchestrator import SupervisorOrchestrator
from utils.CreateLLMAgents import load_llm_agents
from utils.CreateToolAgent import ToolAgent
from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole

# Create simple math tool for demo purposes
async def calculate(user_input: str, **kwargs) -> str:
    """Simple calculator function for demo purposes"""
    try:
        # Extract numbers and operations from the input
        # This is a very simple implementation
        # For a real app, use a proper expression parser
        calculation = user_input.lower().replace('calculate', '').replace('what is', '').strip()
        result = eval(calculation)  # Note: eval is unsafe for production use
        return f"The result of {calculation} is {result}"
    except Exception as e:
        return f"Sorry, I couldn't calculate that. Error: {str(e)}"

# Set up AWS clients
def create_bedrock_client():
    """Create and configure Bedrock client"""
    try:
        # Use environment variables for credentials or default profile
        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get('AWS_REGION')
        )
        return bedrock_runtime
    except Exception as e:
        print(f"Error creating Bedrock client: {str(e)}")
        raise

# Main function
async def main():
    # Constants
    SUPERVISOR_MODEL_ID = os.environ.get('SUPERVISOR_MODEL_ID')
    USER_ID = 'user123'
    SESSION_ID = str(uuid.uuid4())
    MODEL_ID = os.environ.get('MODEL_ID')
    
    # Create Bedrock client
    bedrock_runtime = create_bedrock_client()
    
    # Create supervisor agent
    supervisor_options = BedrockLLMAgentOptions(
        name="supervisor",
        description="The supervisor that coordinates other agents, and give \
        user a unified answer",
        model_id=SUPERVISOR_MODEL_ID,
        client=bedrock_runtime,
        streaming=False
    )
    supervisor_agent = BedrockLLMAgent(supervisor_options)
    
    # Initialize orchestrator
    orchestrator = SupervisorOrchestrator(supervisor_agent)
    
    # Define LLM agent configurations
    llm_agent_configs = [
        {
            "name": "tech_agent",
            "description": "Specialized in technology areas including software development, hardware, AI, cyber security",
            "model_id": MODEL_ID,
            "streaming": False,
        },
        {
            "name": "travel_agent",
            "description": "Helps with travel planning, hotel recommendations, and itinerary creation",
            "model_id": MODEL_ID,
            "streaming": False,
        }
    ]
    
    # Load LLM agents
    load_llm_agents(llm_agent_configs, orchestrator, bedrock_runtime)
    
    # Define and create tool agents
    calculator_agent_config = {
        "name": "calculator",
        "description": "Performs mathematical calculations",
        "tools": [
            {
                "name": "basic_math",
                "description": "Performs basic arithmetic operations",
                "type": "function",
                "module": "__main__",  # Using the calculate function defined in this file
                "function": "calculate",
                "keywords": ["calculate", "compute", "math", "+", "-", "*", "/"]
            }
        ]
    }
    
    # Create and add tool agent
    calculator_agent = ToolAgent(
        calculator_agent_config["name"],
        calculator_agent_config["description"],
        calculator_agent_config["tools"]
    )
    orchestrator.add_agent(calculator_agent)
    
    # Simple CLI loop for interaction
    print("Multi-agent system initialized! Type 'exit' to quit.")
    print(f"Available agents: {orchestrator.list_agents()}")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        print("\nProcessing...")
        response = await orchestrator.route_request(user_input, USER_ID, SESSION_ID)
        
        print(f"\nResponse: {response.output}")
        print(f"(Source: {response.metadata.get('source', 'unknown')})")

if __name__ == "__main__":
    asyncio.run(main())