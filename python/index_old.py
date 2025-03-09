import uuid
import asyncio
from typing import Optional, List, Dict, Any
import urllib.parse
#import multi_agent_orchestrator.agents as agents
#print(dir(agents))
#print(agents._AWS_AVAILABLE)

import boto3
import os
# Set environment variables
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-2'

# Initialize boto3 early
session = boto3.Session(region_name='eu-west-2')
bedrock = session.client('bedrock')  # For listing models
bedrock_runtime = session.client('bedrock-runtime')  # For inference

# client = boto3.client('bedrock', region_name='eu-west-2')
# response = client.list_foundation_models()
# for model in response['modelSummaries']:
#     print(f"Model ID: {model['modelId']}")

# Try this temporarily
""" import inspect
from multi_agent_orchestrator.agents.bedrock_llm_agent import BedrockLLMAgent
print(inspect.signature(BedrockLLMAgent.__init__))
print(inspect.getdoc(BedrockLLMAgent.__init__))

exit() """

from src.orchestrator.CustomOrchestrator import SimpleOrchestrator

import json
import sys
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.agents import ( BedrockLLMAgent, BedrockLLMAgentOptions, AgentResponse, AgentCallbacks )
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole

""" orchestrator = MultiAgentOrchestrator(options=OrchestratorConfig(
    LOG_AGENT_CHAT=True,
    LOG_CLASSIFIER_CHAT=True,
    LOG_CLASSIFIER_RAW_OUTPUT=True,
    LOG_CLASSIFIER_OUTPUT=True,
    LOG_EXECUTION_TIMES=True,
    MAX_RETRIES=3,
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
    MAX_MESSAGE_PAIRS_PER_AGENT=10
))  """
from python.custom_classifier import DirectInvokeClassifier
from utils.BedrockLLMAgentCallbacks import BedrockLLMAgentCallbacks

# Just remove the CLASSIFIER parameter entirely
orchestrator = SimpleOrchestrator(default_agent_name="tech_agent")


        
tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="tech_agent",
    streaming=True,
    description="Specialized in technology areas including software development, hardware, AI. \
        cyber security, and more blockchain, cloud computing, emerging tech innovations, and pricing/costs \
        related to technology products and services.",
    model_id="anthropic.claude-3-haiku-20240307-v1:0", # Ensure exact format
    client=bedrock_runtime,  # Pass the runtime client explicitly
    callbacks=BedrockLLMAgentCallbacks()
))
        
""" tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="tech_agent",
    streaming=True,
    description="Specialized in technology areas including software development, hardware, AI. \
        cyber security, and more blockchain, cloud computing, emerging tech innovations, and pricing/costs \
        related to technology products and services.",
    #model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    callbacks=BedrockLLMAgentCallbacks(),
    client=bedrock_runtime
)) """

""" tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="tech_agent",
    streaming=True,
    description="Specialized in technology areas including software development, hardware, AI. \
        cyber security, and more blockchain, cloud computing, emerging tech innovations, and pricing/costs \
        related to technology products and services.",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    callbacks=BedrockLLMAgentCallbacks(),
    region="eu-west-2"  # Try adding the region explicitly
)) """

orchestrator.add_agent(tech_agent)

async def handle_request(_orchestrator: MultiAgentOrchestrator, _user_input:str, _user_id:str, _session_id: str):
    response: AgentResponse = await _orchestrator.route_request(_user_input, _user_id, _session_id)
    
    
    #print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")  
    if response.streaming:
        print('Response: ', response.output)
    else:
        print('Response: ', response.output)
           

if __name__ == "__main__":
    USER_ID = 'user123'
    SESSION_ID = str(uuid.uuid4())
    print("Welcome to the interactive Multi-agent System! Type 'quit' to exit.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()
        # Run the async function
        asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))