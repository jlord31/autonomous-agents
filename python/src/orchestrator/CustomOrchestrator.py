from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.agents import AgentResponse, Agent
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from dataclasses import dataclass

@dataclass
class SimpleAgentResponseMetadata:
    agent_name: str

class SimpleOrchestrator:
    def __init__(self, default_agent_name):
        self.default_agent_name = default_agent_name
        self.agents = {}  # Initialize an empty agents dictionary
        self.chat_histories = {}  # Store chat histories by session_id
    
    def add_agent(self, agent):
        self.agents[agent.name] = agent
    
    async def route_request(self, user_input, user_id, session_id):
        # Get the default agent
        agent = self.agents.get(self.default_agent_name)
        if not agent:
            raise ValueError(f"Default agent '{self.default_agent_name}' not found")
        
        # Initialize chat history for this session if it doesn't exist
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        
        # Get the chat history for this session
        chat_history = self.chat_histories[session_id]
        
        # Add user message to history
        chat_history.append(ConversationMessage(
            role=ParticipantRole.USER,
            content=[{"text": user_input}]
        ))
        
        # Call the agent with the chat history
        response_message = await agent.process_request(user_input, user_id, session_id, chat_history)
        
        # Extract the text content from the message
        response_text = ""
        if response_message and hasattr(response_message, "content"):
            if isinstance(response_message.content, list) and response_message.content:
                # If it's a list of content blocks
                for content_block in response_message.content:
                    if isinstance(content_block, dict) and "text" in content_block:
                        response_text += content_block["text"]
            
        # Add agent's response to history
        chat_history.append(ConversationMessage(
            role=ParticipantRole.ASSISTANT,
            content=[{"text": response_text}]
        ))
        
        # Create a standardized response object that matches the expected interface
        return AgentResponse(
            output=response_text,
            metadata=SimpleAgentResponseMetadata(agent_name=agent.name),
            streaming=False
        )