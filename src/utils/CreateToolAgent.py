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
    
    def _extract_expression(self, user_input: str, keywords: List[str] = None) -> str:
        """Extract the calculation expression from user input"""
        import re
        
        # Clean the input by removing keywords
        cleaned_input = user_input
        if keywords:
            for keyword in keywords:
                cleaned_input = cleaned_input.replace(keyword, ' ')
        
        # Look for patterns like (2+8)/2 + 5 or other math expressions
        math_pattern = re.compile(r'[\(]?[\d\+\-\*\/\.\(\)\s\^\%]+[\)]?')
        matches = math_pattern.findall(cleaned_input)
        
        if matches:
            # Use the longest match as it's likely the full expression
            return max(matches, key=len).strip()
        
        # If no matches found, return a cleaned version of original input
        return re.sub(r'[^\w\s\.\,\+\-\*\/\(\)\^\%]', '', cleaned_input).strip()
    
    async def _execute_tool(self, tool: Dict, user_input: str) -> str:
        """Execute the specified tool"""
        try:
            # Get module and function information
            module_name = tool.get('module')
            function_name = tool.get('function')
            
            # Import the module dynamically
            module = importlib.import_module(module_name)
            
            # IMPORTANT FIX: Extract class name from module path
            class_name = module_name.split('.')[-1]
            
            # Get the class from the module - if class_name is 'CalculatorTool', look for that class
            if hasattr(module, class_name):
                tool_class = getattr(module, class_name)
            else:
                # Try other common patterns
                if '.' in module_name:
                    # For paths like 'tools.CalculatorTool', look for class 'CalculatorTool'
                    simple_name = module_name.split('.')[-1]
                    if hasattr(module, simple_name):
                        tool_class = getattr(module, simple_name)
                    else:
                        # Look for the actual module components
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if hasattr(attr, function_name) and callable(getattr(attr, function_name)):
                                tool_class = attr
                                break
                        else:
                            # If we get here, we couldn't find the class
                            print(f"Debug: Module contents: {dir(module)}")
                            raise AttributeError(f"Could not find appropriate class in {module_name}")
                else:
                    raise AttributeError(f"Could not find class in {module_name}")
            
            # Prepare parameters
            params = {
                "query": user_input,
                "expression": self._extract_expression(user_input, tool.get('keywords', []))
            }
            
            # DEBUG PRINT
            print(f"Found class {tool_class.__name__} in module {module_name}")
            print(f"Calling {function_name} on {tool_class.__name__}")
            print(f"User input: '{user_input}'")
            print(f"Extracted expression: '{params['expression']}'")  
            # Call the run method on the class
            return await getattr(tool_class, function_name)(params)
            
        except ImportError as e:
            return f"Error importing module '{module_name}': {str(e)}"
        except AttributeError as e:
            return f"Error accessing class or method: {str(e)}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"


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