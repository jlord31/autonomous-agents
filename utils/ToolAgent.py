import aiohttp
import importlib
from typing import Dict, List
from multi_agent_orchestrator.agents import (BedrockLLMAgent, BedrockLLMAgentOptions, AgentResponse, Agent)
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole

class ToolAgent(Agent):
    """An agent that's just a wrapper around one or more tools"""
    
    def __init__(self, name: str, description: str, tools: List[Dict]):
        self.name = name
        self.description = description
        self.tools = tools
    
    async def process_request(self, user_input: str, user_id: str, session_id: str, 
                             history: List[ConversationMessage] = None) -> AgentResponse:
        """Process request using the appropriate tool"""
        # Parse the request to determine which tool to use
        tool_name = self._determine_tool(user_input)
        tool = next((t for t in self.tools if t.get('name') == tool_name), None)
        
        if not tool:
            return AgentResponse(
                output=f"No suitable tool found for this request in agent {self.name}",
                metadata={"error": "no_suitable_tool"},
                streaming=False
            )
            
        try:
            # Call the tool function
            result = await self._execute_tool(tool, user_input)
            return AgentResponse(
                output=result,
                metadata={"tool": tool_name},
                streaming=False
            )
        except Exception as e:
            return AgentResponse(
                output=f"Error executing tool {tool_name}: {str(e)}",
                metadata={"error": str(e)},
                streaming=False
            )
    
    def _determine_tool(self, user_input: str) -> str:
        """Simple tool selection based on keywords in the input"""
        # This is a simple implementation - you might want more sophisticated matching
        for tool in self.tools:
            keywords = tool.get('keywords', [])
            if any(keyword.lower() in user_input.lower() for keyword in keywords):
                return tool.get('name')
        
        # Default to first tool if no match
        return self.tools[0].get('name') if self.tools else None
    
    async def _execute_tool(self, tool: Dict, user_input: str) -> str:
        """Execute the specified tool"""
        tool_type = tool.get('type')
        
        if tool_type == 'function':
            # Import and call the function dynamically
            module_name = tool.get('module')
            function_name = tool.get('function')
            

            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
            
            # Call the function
            args = tool.get('args', {})
            return await function(user_input, **args) if callable(function) else "Function not callable"
            
        elif tool_type == 'api':
            # Make API call
            
            url = tool.get('url').format(input=user_input)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        return f"API error: {response.status}"
        
        else:
            return f"Unknown tool type: {tool_type}"


# Example tool function
""" {
  "name": "calculator_agent",
  "description": "Performs mathematical calculations",
  "type": "tool",
  "tools": [
    {
      "name": "basic_math",
      "description": "Performs basic arithmetic",
      "type": "function",
      "module": "math_tools",
      "function": "calculate",
      "keywords": ["calculate", "sum", "add", "subtract", "multiply", "divide"]
    },
    {
      "name": "unit_converter",
      "description": "Converts between different units",
      "type": "api",
      "url": "https://api.example.com/convert?query={input}",
      "keywords": ["convert", "unit", "meter", "feet", "celsius", "fahrenheit"]
    }
  ]
} """