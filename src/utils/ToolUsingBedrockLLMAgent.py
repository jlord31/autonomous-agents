from typing import Dict, List, Any
from multi_agent_orchestrator.agents import BedrockLLMAgent, BedrockLLMAgentOptions
import importlib
import json

class ToolUsingBedrockLLMAgent(BedrockLLMAgent):
    """Extension of BedrockLLMAgent that can use tools"""
    
    async def _process_tool_calls(self, response_text: str):
        """Process any tool calls in the response"""
        # Look for tool call patterns
        tool_pattern = r"TOOL_CALL\[(.*?)\]TOOL_INPUT\[(.*?)\]TOOL_CALL_END"
        import re
        
        tool_calls = re.findall(tool_pattern, response_text, re.DOTALL)
        new_response = response_text
        
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
                        params = json.loads(tool_input)
                    except:
                        # If not valid JSON, use as raw input
                        params = {"input": tool_input}
                    
                    # Import the module and run the tool
                    module_name = tool.get('module')
                    function_name = tool.get('function')
                    
                    module = importlib.import_module(module_name)
                    tool_class = getattr(module, module_name.split('.')[-1])
                    tool_result = await getattr(tool_class, function_name)(params)
                    
                except Exception as e:
                    tool_result = f"Error executing tool {tool_name}: {str(e)}"
            
            # Replace the tool call with the result
            tool_call_text = f"TOOL_CALL[{tool_name}]TOOL_INPUT[{tool_input}]TOOL_CALL_END"
            new_response = new_response.replace(tool_call_text, f"TOOL_RESULT: {tool_result}")
        
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