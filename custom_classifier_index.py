import uuid
import asyncio
from typing import Optional, List, Dict, Any
import boto3
# Initialize boto3 early
session = boto3.Session(region_name='eu-west-2')
bedrock = session.client('bedrock')  # For listing models
bedrock_runtime = session.client('bedrock-runtime')  # For inference

import json
import sys
import os
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.agents import ( 
                                            BedrockLLMAgent, 
                                            LexBotAgent, 
                                            LexBotAgentOptions, 
                                            BedrockLLMAgentOptions, 
                                            AgentResponse, 
                                            AgentCallbacks 
)
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole

from custom_classifier import InvokeModelClassifier
from utils.BedrockLLMAgentCallbacks import BedrockLLMAgentCallbacks

from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")

# Initialize classifier
custom_classifier = InvokeModelClassifier(
    client=bedrock_runtime,
    model_id=MODEL_ID
)

# Initialize orchestrator with custom classifier
orchestrator = MultiAgentOrchestrator(
    options=OrchestratorConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        MAX_MESSAGE_PAIRS_PER_AGENT=10
    ),
    classifier=custom_classifier  # Using custom classifier
)

# Add your agents
tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="tech_agent",
    streaming=True,
    description="Specialized in technology areas including software development, hardware, AI. \
        cyber security, and more blockchain, cloud computing, emerging tech innovations, and pricing/costs \
        related to technology products and services.",
    model_id=MODEL_ID,
    client=bedrock_runtime,
    callbacks=BedrockLLMAgentCallbacks()
))

travel_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="travel_agent",
    streaming=True,
    description="Helps users book and manage their flight reservations",
    model_id=MODEL_ID,
    client=bedrock_runtime,
    callbacks=BedrockLLMAgentCallbacks()
))

orchestrator.add_agent(tech_agent)
orchestrator.add_agent(travel_agent)

async def handle_request(_orchestrator: MultiAgentOrchestrator, _user_input:str, _user_id:str, _session_id: str):
    response: AgentResponse = await _orchestrator.route_request(_user_input, _user_id, _session_id)
    
    
    #print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")  

    """ if response.streaming:
        print('Response: ', response.output)
    else:
        print('Response: ', response.output) """
    
    # Fix this part to properly extract text from ConversationMessage
    if hasattr(response.output, 'content'):
        # It's a ConversationMessage object
        if isinstance(response.output.content, list) and response.output.content:
            for content_block in response.output.content:
                if isinstance(content_block, dict) and 'text' in content_block:
                    print(f"Response: {content_block['text']}")
    else:
        # It's a string or other type
        print(f"Response: {response.output}")
           

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