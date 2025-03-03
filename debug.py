import uuid
import asyncio
from typing import Optional, List, Dict, Any
import urllib.parse
import boto3
import os
import inspect
import logging
import json

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('botocore').setLevel(logging.DEBUG)

# Set environment variables
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-2'

# Initialize boto3 early
session = boto3.Session(region_name='eu-west-2')
bedrock = session.client('bedrock')  # For listing models
bedrock_runtime = session.client('bedrock-runtime')  # For inference

# Verify direct AWS access works
try:
    response = bedrock.list_foundation_models()
    print("AWS Bedrock API direct access works!")
    print(f"Found {len(response['modelSummaries'])} models")
except Exception as e:
    print(f"AWS Bedrock API direct access failed: {e}")

# Import the agent class and examine it
from multi_agent_orchestrator.agents.bedrock_llm_agent import BedrockLLMAgent
print(inspect.signature(BedrockLLMAgent.__init__))
print(inspect.getdoc(BedrockLLMAgent.__init__))

# Look at the actual implementation
source_code = inspect.getsource(BedrockLLMAgent)
print("\nBedrockLLMAgent source code snippet:")
print("\n".join(source_code.split("\n")[:30]))  # Print first 30 lines

# Attempt to make a direct call using bedrock-runtime client
try:
    # For Anthropic Claude models, the request format is different
    anthropic_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ]
    }
    
    test_response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps(anthropic_payload)
    )
    
    # Parse and print response
    response_body = json.loads(test_response.get('body').read().decode())
    print("Direct invoke_model call succeeded!")
    print(f"Response content: {response_body}")
except Exception as e:
    print(f"Direct invoke_model call failed: {e}")
    import traceback
    traceback.print_exc()

# Check orchestrator imports and configuration
try:
    from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
    from multi_agent_orchestrator.agents import AgentCallbacks
    
    # Print what converse operations are being used
    orchestrator_source = inspect.getsource(MultiAgentOrchestrator)
    if "converse" in orchestrator_source:
        print("\nFound 'converse' in orchestrator source code!")
        # Extract lines with "converse" in them
        converse_lines = [line for line in orchestrator_source.split("\n") 
                          if "converse" in line.lower()]
        print("\n".join(converse_lines[:10]))  # Show first 10 occurrences
except Exception as e:
    print(f"Error inspecting orchestrator: {e}")

# Continue with the rest of your code



