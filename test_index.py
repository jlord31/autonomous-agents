import uuid
import asyncio
from typing import Optional, List, Dict, Any
import urllib.parse
import boto3
import os
# Set environment variables
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-2'

# Initialize boto3 early
session = boto3.Session(region_name='eu-west-2')
bedrock = session.client('bedrock')  # For listing models
bedrock_runtime = session.client('bedrock-runtime')  # For inference

from CustomOrchestrator import SimpleOrchestrator

import json
import sys
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.agents import ( BedrockLLMAgent, BedrockLLMAgentOptions, AgentResponse, AgentCallbacks )
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole

orchestrator = MultiAgentOrchestrator(options=OrchestratorConfig(
    LOG_AGENT_CHAT=True,
    LOG_CLASSIFIER_CHAT=True,
    LOG_CLASSIFIER_RAW_OUTPUT=True,
    LOG_CLASSIFIER_OUTPUT=True,
    LOG_EXECUTION_TIMES=True,
    MAX_RETRIES=3,
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
    MAX_MESSAGE_PAIRS_PER_AGENT=10
)) 

class BedrockLLMAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        # handle response streaming here
        print(token, end='', flush=True)
        
tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="tech_agent",
    streaming=True,
    description="Specialized in technology areas including software development, hardware, AI. \
        cyber security, and more blockchain, cloud computing, emerging tech innovations, and pricing/costs \
        related to technology products and services.",
    #model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
    #model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_id="amazon.titan-text-lite-v1",
    callbacks=BedrockLLMAgentCallbacks(),
    client=bedrock_runtime
))

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