import uuid
import os
import asyncio
import boto3
from typing import Dict, List
import dotenv
from dotenv import load_dotenv

# Import our custom components
from orchestrator.supervisor_orchestrator import SupervisorOrchestrator
from utils.CreateLLMAgents import load_llm_agents
from utils.CreateToolAgent import ToolAgent
from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from tools.registry.index import get_tool_configs

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
    
    # Get all tool configs
    all_tools = get_tool_configs()
    email_tool = next((t for t in all_tools if t["name"] == "send_email"), None)
    calculator_tools = [t for t in all_tools if t["name"] == "calculator"]


    # Define LLM agent configurations
    # Define LLM agent configurations - attach tools to LLM agents
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
        },
        {
            "name": "math_assistant",  
            "description": "Performs mathematical calculations including arithmetic operations, equations, and unit conversions",
            "model_id": MODEL_ID,
            "streaming": False,
            "tools": calculator_tools  
        },
        {
            "name": "email_assistant",
            "description": "Helps you compose and send emails, manage communication, and draft professional correspondence",
            "model_id": MODEL_ID,
            "streaming": False,
            "tools": [email_tool] if email_tool else []
        }
    ]
    
    # Load LLM agents
    load_llm_agents(llm_agent_configs, orchestrator, bedrock_runtime)
    
    # Define and create tool agents
    
    # if calculator_tools:
    #     calculator_agent = ToolAgent(
    #         name="calculator",
    #         description="Performs mathematical calculations including arithmetic operations, equations, and unit conversions",
    #         tools=calculator_tools
    #     )
    #     orchestrator.add_agent(calculator_agent)
    #     print(f"Added agent: {calculator_agent.name}")
        
    #     # Simple CLI loop for interaction
    #     print("Multi-agent system initialized! Type 'exit' to quit.")
    #     print(f"Available agents: {orchestrator.list_agents()}")
    
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