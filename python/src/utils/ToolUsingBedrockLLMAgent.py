from typing import Dict, List, Any
from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
import importlib
import json
import re
import asyncio

class ToolUsingBedrockLLMAgent(BedrockLLMAgent):
    """Extension of BedrockLLMAgent that can use tools"""
    
    async def _process_tool_calls(self, response_text: str):
        """Process any tool calls in the response"""
        # Look for tool call patterns
        print(f"Checking for tool calls in: {response_text[:100]}...")
        
            # Add more flexible pattern matching - LLMs often add spaces or format inconsistently
        # Multiple patterns to match different ways the LLM might format tool calls
        tool_patterns = [
            r'TOOL_CALL\[(.*?)\]TOOL_INPUT\[(.*?)\]TOOL_CALL_END',  # Standard format
            r'TOOL_CALL\[(.*?)\]TOOL_INPUT\s*(\{.*?\})\s*TOOL_CALL_END',  # JSON with spaces
            r'```(?:json)?\s*TOOL_CALL\[(.*?)\]TOOL_INPUT\[(.*?)\]```',  # With code blocks
            r'Using tool:\s*(.*?)\nWith parameters:\s*(\{.*?\})',  # Natural language format
        ]
        
        # Try each pattern
        tool_calls = []
        for pattern in tool_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                print(f"Found tool calls with pattern: {pattern}")
                tool_calls.extend(matches)
                break

        # Process found tool calls
        if tool_calls:
            print(f"Found {len(tool_calls)} tool calls:")
            
            # In _process_tool_calls method:
        for tool_name, tool_input in tool_calls:
            tool_name = tool_name.strip()
            
            # Find the tool
            tool = next((t for t in self.tools if t.get("name") == tool_name), None)
            
            if not tool:
                tool_result = f"Error: Tool '{tool_name}' not found"
            else:
                try:
                    # Parse the tool input as parameters
                    try:
                        # Clean up the tool_input string - remove any markdown formatting
                        tool_input = tool_input.strip()
                        if tool_input.startswith('```') and tool_input.endswith('```'):
                            tool_input = tool_input[3:-3].strip()
                        
                        # Parse as JSON
                        params = json.loads(tool_input)
                        print(f"Parsed tool parameters: {params}")
                        
                        # Dynamically import the tool module
                        module_name = tool.get("module")
                        function_name = tool.get("function")
                        
                        if not module_name or not function_name:
                            raise ValueError(f"Missing module or function for tool {tool_name}")
                        
                        # Import the module
                        module = importlib.import_module(module_name)
                        function = getattr(module, function_name)
                        
                        # Call the function
                        print(f"Calling function {function_name} with params {params}")
                        if asyncio.iscoroutinefunction(function):
                            tool_result = await function(**params)
                        else:
                            tool_result = function(**params)
                        
                        print(f"Tool execution result: {tool_result}")
                    except json.JSONDecodeError:
                        tool_result = f"Error: Invalid tool input format. Expected JSON."
                except Exception as e:
                    tool_result = f"Error executing tool: {str(e)}"
            
            return new_response

    async def process_request(self, user_input, user_id, session_id, history=None):
        """Override process_request to handle tool calling"""
        response = await super().process_request(user_input, user_id, session_id, history)
        
        # Process any tool calls in the response
        if hasattr(response, 'content'):
            for i, content_block in enumerate(response.content):
                if isinstance(content_block, dict) and 'text' in content_block:
                    response.content[i]['text'] = await self._process_tool_calls(content_block['text'])
        elif hasattr(response, 'output'):
            response.output = await self._process_tool_calls(response.output)
        
        return response