from typing import List, Dict, Any, Optional
import json
import boto3
from multi_agent_orchestrator.classifiers import Classifier, ClassifierResult
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.agents import Agent

class InvokeModelClassifier(Classifier):
    def __init__(self, client, model_id):
        super().__init__()
        self.client = client
        self.model_id = model_id
        self.agents = {}
        
    def set_agents(self, agents: Dict[str, Agent]) -> None:
        self.agents = agents
    
    # Add the missing process_request method
    async def process_request(self, user_input: str, user_id: str, session_id: str, chat_history: List[ConversationMessage] = None) -> ClassifierResult:
        """Required implementation of the abstract method from Classifier base class"""
        # Simply delegate to the classify method
        return await self.classify(user_input, chat_history)
    
    async def classify(self, input_text: str, chat_history: List[ConversationMessage] = None) -> ClassifierResult:
        """Classify using direct invoke_model instead of Converse"""
        # If only one agent, just return it
        if len(self.agents) == 1:
            agent = next(iter(self.agents.values()))
            return ClassifierResult(selected_agent=agent, confidence=1.0)
        
        # Get agent descriptions for classification
        agent_options = "\n".join([
            f"{agent.name}: {agent.description}"
            for agent in self.agents.values()
        ])
    
        # Build a better prompt for the model
        prompt = [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an agent router that determines which specialized agent should handle a user request.

                        USER REQUEST: "{input_text}"

                        AVAILABLE AGENTS:
                        {agent_options}

                        INSTRUCTIONS:
                        1. Analyze the user request carefully
                        2. Determine which agent's expertise best matches the request
                        3. Respond with ONLY the exact name of the selected agent, nothing else

                        For example, if the request is about flight bookings, respond with: travel_agent
                        If the request is about software development, respond with: tech_agent"""
                    }
                ]
            }
        ]
    
        try:
            # Direct invoke model 
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 30,  # Short response needed
                    "messages": prompt,
                    "temperature": 0.1  # Low temperature for more deterministic response
                })
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read().decode())
            agent_name = response_body['content'][0]['text'].strip()
            
            print(f"Classifier selected: '{agent_name}'")
            
            # Find the exact matching agent by name
            for agent_id, agent in self.agents.items():
                if agent.name.lower() == agent_name.lower():
                    return ClassifierResult(selected_agent=agent, confidence=0.95)
                
            # If no exact match but contains name, use that
            for agent_id, agent in self.agents.items():
                if agent.name.lower() in agent_name.lower():
                    return ClassifierResult(selected_agent=agent, confidence=0.8)
            
            # If still no match, use default behavior
            print(f"No agent match found. Model response was: '{agent_name}'")
            default_agent = next(iter(self.agents.values()))
            return ClassifierResult(selected_agent=default_agent, confidence=0.1)
                
        except Exception as e:
            print(f"Classification error: {str(e)}")
            default_agent = next(iter(self.agents.values())) 
            return ClassifierResult(selected_agent=default_agent, confidence=0.1)