import uuid
import asyncio
import boto3
import os
import json
import sys

# Set up AWS clients
session = boto3.Session(region_name='eu-west-2')
bedrock_runtime = session.client('bedrock-runtime')

from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions, AgentCallbacks

class BedrockLLMAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        print(token, end='', flush=True)

# Create your agent
agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="tech_agent",
    streaming=True,
    description="Specialized in technology areas...",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    client=bedrock_runtime,
    callbacks=BedrockLLMAgentCallbacks()
))

# Handle requests directly
async def handle_request(user_input, user_id, session_id):
    response = await agent.process_request(user_input, user_id, session_id)
    return response

# Main loop
if __name__ == "__main__":
    USER_ID = 'user123'
    SESSION_ID = str(uuid.uuid4())
    print("Welcome to the AI Assistant! Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()
        response = asyncio.run(handle_request(user_input, USER_ID, SESSION_ID))
        print("\nResponse:", response.output)