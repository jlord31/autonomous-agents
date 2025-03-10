# import uuid
# import os
# import asyncio
# import boto3
# import uvicorn
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Import api for FastAPI
# from api import app

# # For direct console usage (optional)
# async def console_mode():
#     """Run the orchestrator in console mode for testing"""
#     from orchestrator.supervisor_orchestrator import SupervisorOrchestrator
#     from utils.CreateLLMAgents import load_llm_agents
#     from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
#     from tools.registry.index import get_tool_configs
    
#     # Create components
#     bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_REGION'))
#     supervisor_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
#         name="supervisor", 
#         model_id=os.environ.get('SUPERVISOR_MODEL_ID'),
#         client=bedrock_runtime
#     ))
#     orchestrator = SupervisorOrchestrator(supervisor_agent)
    
#     # Load tools and agents
#     all_tools = get_tool_configs()
#     email_tool = next((t for t in all_tools if t["name"] == "send_email"), None)
#     calculator_tools = [t for t in all_tools if t["name"] == "calculator"]
    
#     llm_agent_configs = [
#         {"name": "tech_agent", "description": "Tech specialist", "model_id": os.environ.get('MODEL_ID')},
#         {"name": "travel_agent", "description": "Travel specialist", "model_id": os.environ.get('MODEL_ID')},
#         {"name": "math_assistant", "description": "Math specialist", "model_id": os.environ.get('MODEL_ID'), "tools": calculator_tools},
#         {"name": "email_assistant", "description": "Email specialist", "model_id": os.environ.get('MODEL_ID'), "tools": [email_tool] if email_tool else []}
#     ]
#     load_llm_agents(llm_agent_configs, orchestrator, bedrock_runtime)
    
#     # Interactive loop
#     user_id = "console_user"
#     session_id = str(uuid.uuid4())
    
#     print("Interactive mode - type 'exit' to quit")
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() == 'exit':
#             break
            
#         print("Processing...")
#         response = await orchestrator.route_request(user_input, user_id, session_id)
#         print(f"\nResponse: {response.output}")
#         print(f"(Source: {response.metadata.get('source', 'unknown')})")

# if __name__ == "__main__":
#     # Check if we should run in API mode or console mode
#     api_mode = os.environ.get('API_MODE', 'true').lower() == 'true'
    
#     if api_mode:
#         # Start the FastAPI server
#         print("Starting in API mode")
#         uvicorn.run(app, host="0.0.0.0", port=8000)
#     else:
#         # Run in console mode
#         print("Starting in console mode")
#         asyncio.run(console_mode())

import uuid
import os
import asyncio
import boto3
import uvicorn
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.getLevelName(os.environ.get('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import api for FastAPI
from api import app

# For direct console usage (testing/development)
async def console_mode():
    """Run the orchestrator in console mode for testing"""
    from orchestrator.supervisor_orchestrator import SupervisorOrchestrator
    from utils.CreateLLMAgents import load_llm_agents
    from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
    from tools.registry.index import get_tool_configs
    
    # Create components
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime', 
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )
    
    supervisor_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="supervisor", 
        model_id=os.environ.get('SUPERVISOR_MODEL_ID'),
        client=bedrock_runtime
    ))
    orchestrator = SupervisorOrchestrator(supervisor_agent)
    
    # Load tools and agents
    all_tools = get_tool_configs()
    email_tool = next((t for t in all_tools if t["name"] == "send_email"), None)
    calculator_tools = [t for t in all_tools if t["name"] == "calculator"]
    
    llm_agent_configs = [
        {"name": "tech_agent", "description": "Tech specialist", "model_id": os.environ.get('MODEL_ID')},
        {"name": "travel_agent", "description": "Travel specialist", "model_id": os.environ.get('MODEL_ID')},
        {"name": "math_assistant", "description": "Math specialist", "model_id": os.environ.get('MODEL_ID'), "tools": calculator_tools},
        {"name": "email_assistant", "description": "Email specialist", "model_id": os.environ.get('MODEL_ID'), "tools": [email_tool] if email_tool else []}
    ]
    load_llm_agents(llm_agent_configs, orchestrator, bedrock_runtime)
    
    # Interactive loop
    user_id = "console_user"
    session_id = str(uuid.uuid4())
    
    print("\n===== Multi-Agent Orchestrator Console =====")
    print("Type 'exit' to quit")
    print("===========================================\n")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        print("\nProcessing...")
        try:
            response = await orchestrator.route_request(user_input, user_id, session_id)
            print(f"\nResponse: {response.output}")
            print(f"(Source: {response.metadata.get('source', 'unknown')})")
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    # Check if we should run in API mode or console mode
    api_mode = os.environ.get('API_MODE', 'true').lower() == 'true'
    
    if api_mode:
        # Start the FastAPI server
        logger.info("Starting in API mode")
        port = int(os.environ.get('PORT', 8000))
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level=os.environ.get('LOG_LEVEL', 'info').lower()
        )
    else:
        # Run in console mode
        logger.info("Starting in console mode")
        asyncio.run(console_mode())