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
            "system_prompt": "You are a math assistant with access to calculator tools. \
                IMPORTANT: To perform calculations, you MUST use the provided tools. \
                Never calculate manually. Format tool calls exactly as: \
                TOOL_CALL[calculator]TOOL_INPUT{\"operation\": \"multiply\", \"numbers\": [28749, 1823]}TOOL_CALL_END \
                Available tool operations: add, subtract, multiply, divide \
            ",
            "model_id": MODEL_ID,
            "streaming": False,
            "tools": calculator_tools  
        },
        {
            "name": "email_assistant",
            "description": "Helps you compose and send emails, manage communication, and draft professional correspondence",
            "system_prompt": """You are an email assistant that can help users compose and send emails.

                IMPORTANT: You have access to a send_email tool. To send emails, you MUST use this tool.
                DO NOT pretend to send emails without using the tool.

                When ready to send an email, use EXACTLY this format:
                TOOL_CALL[send_email]TOOL_INPUT{"to_email": "recipient@example.com", "subject": "Subject line", "body": "Email content"}TOOL_CALL_END

                Steps for email handling:
                1. Help user compose the email (recipient, subject, body)
                2. Show the draft and confirm with user before sending
                3. ONLY after user confirmation, call the send_email tool with proper parameters
                4. Report the result of the sending operation to the user

                Example of correct tool usage:
                User: Send an email to joe@example.com with subject "Meeting"
                You: Let me draft that for you. How about this? [Draft email]
                User: Looks good, please send it
                You: I'll send it now. 
                TOOL_CALL[send_email]TOOL_INPUT{"to_email": "joe@example.com", "subject": "Meeting", "body": "Dear Joe..."}TOOL_CALL_END
                Email sent successfully!

                Never claim you have sent an email if you haven't used the send_email tool.
            """,
                "model_id": MODEL_ID,
                "tools": [{"name": "send_email", "module": "tools.email", "function": "send_email"}]
            
        }
    ]
    
    # Load LLM agents
    load_llm_agents(llm_agent_configs, orchestrator, bedrock_runtime)
    
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