import json
import re
import asyncio
from typing import Dict, List, Any, Optional
from multi_agent_orchestrator.agents import Agent, AgentResponse, BedrockLLMAgent
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole

class SupervisorOrchestrator:
    def __init__(self, supervisor_agent: BedrockLLMAgent):
        """Initialize with just the supervisor agent - other agents can be added dynamically"""
        self.supervisor = supervisor_agent
        self.agents = {}  # name -> agent
        self.chat_histories = {}  # session_id -> conversation history
        self.agent_histories = {}  # session_id -> {agent_name -> conversation history}
    
    def add_agent(self, agent: Agent) -> None:
        """Add a specialist agent to the orchestrator"""
        self.agents[agent.name] = agent
        print(f"Added agent: {agent.name}")
    
    def list_agents(self) -> List[str]:
        """List all available specialist agents"""
        return list(self.agents.keys())
    
    def _get_history(self, session_id: str) -> List[ConversationMessage]:
        """Get conversation history for a session"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        return self.chat_histories[session_id]
    
    def _get_agent_history(self, session_id: str, agent_name: str) -> List[ConversationMessage]:
        """Get agent-specific conversation history"""
        if session_id not in self.agent_histories:
            self.agent_histories[session_id] = {}
        if agent_name not in self.agent_histories[session_id]:
            self.agent_histories[session_id][agent_name] = []
        return self.agent_histories[session_id][agent_name]
    
    def _create_agent_descriptions(self) -> str:
        """Create a description of all available agents and their tools"""
        descriptions = []
        
        for agent_name, agent in self.agents.items():
            description = f"- {agent_name}: {agent.description}"
            
            # Add tool information if available
            if hasattr(agent, 'tools') and agent.tools:
                tools_desc = ", ".join([
                    f"{tool.get('name')}: {tool.get('description')}" 
                    for tool in agent.tools
                ])
                description += f" (Tools: {tools_desc})"
                
            descriptions.append(description)
            
        return "\n".join(descriptions)
        
    def _parse_supervisor_plan(self, response_text: str) -> Dict[str, Any]:
        """Extract the execution plan from supervisor response"""
        try:
            # Look for JSON between ``` markers
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                plan_json = json_match.group(1)
                return json.loads(plan_json)
            
            # If no structured JSON found, use simple agent name extraction
            plan = {"reasoning": "Extracted from text response", "actions": []}
            
            # Look for any agent names mentioned in the response
            for agent_name in self.agents.keys():
                if agent_name.lower() in response_text.lower():
                    # Found an agent reference, extract surrounding context as query
                    agent_pos = response_text.lower().find(agent_name.lower())
                    context_start = max(0, agent_pos - 100)
                    context_end = min(len(response_text), agent_pos + 100)
                    context = response_text[context_start:context_end]
                    
                    plan["actions"].append({
                        "type": "call_specialist",
                        "agent": agent_name,
                        "query": context  # Use surrounding context as query
                    })
            
            # If we didn't find any agent references but have text, 
            # default to assuming the supervisor is giving a direct answer
            if not plan["actions"] and response_text.strip():
                plan["actions"].append({
                    "type": "supervisor_direct_response",
                    "response": response_text
                })
                
            return plan
        except Exception as e:
            print(f"Error parsing supervisor plan: {str(e)}")
            return {"reasoning": "Error parsing plan", "actions": []}
    
    def _extract_response_text(self, response: Any) -> str:
        """Helper to extract text from various response types"""
        if hasattr(response, 'content') and isinstance(response.content, list):
            return "".join([
                content_block.get("text", "") 
                for content_block in response.content 
                if isinstance(content_block, dict) and "text" in content_block
            ])
        elif hasattr(response, 'output'):
            return response.output
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    async def _process_agent_request(self, agent_name, query, user_id, session_id, output_var=None):
        """Process a request to a specialist agent for parallel execution"""
        agent = self.agents[agent_name]
        agent_history = self._get_agent_history(session_id, agent_name)
        
        # Add query to agent history
        agent_history.append(ConversationMessage(
            role=ParticipantRole.USER,
            content=[{"text": query}]
        ))
        
        try:
            print(f"Calling specialist agent (parallel): {agent_name}")
            response = await agent.process_request(query, user_id, session_id, agent_history)
            response_text = self._extract_response_text(response)
            
            # Create response data
            response_data = {
                'agent': agent_name,
                'query': query,
                'response': response_text
            }
            
            # Update agent history
            agent_history.append(ConversationMessage(
                role=ParticipantRole.ASSISTANT,
                content=[{"text": response_text}]
            ))
            
            return {
                'response_data': response_data,
                'response_text': response_text,
                'output_var': output_var
            }
        except Exception as e:
            print(f"Error calling agent {agent_name}: {str(e)}")
            return {
                'response_data': {
                    'agent': agent_name,
                    'query': query,
                    'error': str(e)
                },
                'output_var': output_var
            }
        
    async def route_request(self, user_input: str, user_id: str, session_id: str) -> AgentResponse:
        """Process user request through the supervisor architecture"""
        # Get session history
        history = self._get_history(session_id)
        
        # Add user input to history
        history.append(ConversationMessage(
            role=ParticipantRole.USER,
            content=[{"text": user_input}]
        ))
        
        # Generate dynamic agent descriptions
        agent_descriptions = self._create_agent_descriptions()
        
        # Step 1: Create the non-variable parts of the message
        json_template_option_a = r"""```json
        {
            "reasoning": "Your reasoning about the request",
            "actions": [
                {
                    "type": "call_specialist",
                    "agent": "agent1",
                    "query": "Initial query",
                    "step": 1,
                    "output_var": "result1"
                },
                {
                    "type": "parallel_group",
                    "step": 2,
                    "actions": [
                        {
                            "agent": "agent2",
                            "query": "Process part of {{result1}}",
                            "output_var": "result2a"
                        },
                        {
                            "agent": "agent3",
                            "query": "Process another part of {{result1}}",
                            "output_var": "result2b"
                        }
                    ],
                    "depends_on": ["result1"]
                },
                {
                    "type": "condition",
                    "step": 3,
                    "condition": "{{result2a}} contains 'error'",
                    "if_true": {
                        "agent": "error_handler",
                        "query": "Handle this error: {{result2a}}"
                    },
                    "if_false": {
                        "agent": "agent4",
                        "query": "Continue with {{result2a}} and {{result2b}}",
                        "output_var": "result3"
                    },
                    "depends_on": ["result2a", "result2b"]
                }
            ]
        }
        ```"""
        
        # Step 1: Send request to supervisor with planning instructions
        planning_input = f"""TASK: Determine how to handle this user request.

            USER REQUEST: {user_input}

            AVAILABLE SPECIALIST AGENTS:
            {agent_descriptions}

            INSTRUCTIONS:
            1. Analyze the user request
            2. Decide which specialist agent(s) should handle this request
            3. Provide your plan as valid JSON with the following format (make sure to include all commas between properties):
    
            Option A - If you need specialist agents:
            {json_template_option_a}

            ```json
                Option B - If you can handle directly:
                
                {{
                    "reasoning": "Your reasoning about handling directly",
                    "actions": [
                        {{
                        "type": "supervisor_direct_response",
                        "response": "Your direct response to the user"
                        }}
                    ]
                }}
            ```
        """
        
        # Send to supervisor
        planning_response = await self.supervisor.process_request(
            planning_input, user_id, session_id, history
        )

        # Extract planning response text
        planning_text = self._extract_response_text(planning_response)
        
        print(f"RAW SUPERVISOR RESPONSE:\n{planning_text}")
        
        # Parse the plan
        plan = self._parse_supervisor_plan(planning_text)
        print(f"Supervisor plan: {json.dumps(plan, indent=2)}")

        # Step 2: Execute the plan
        specialist_responses = []
        direct_response = None
        intermediate_results = {} 

        for action in plan.get('actions', []):
            action_type = action.get('type')
        
            # Handle direct response from supervisor
            if action_type == "supervisor_direct_response":
                direct_response = action.get('response', '')
                continue
                
            # Handle specialist agent calls
            elif action_type == "call_specialist":
                agent_name = action.get('agent')
                query = action.get('query', user_input)
                
                if agent_name in self.agents:
                    # Get the agent and its history
                    agent = self.agents[agent_name]
                    agent_history = self._get_agent_history(session_id, agent_name)
                    
                    # Add query to agent history
                    agent_history.append(ConversationMessage(
                        role=ParticipantRole.USER,
                        content=[{"text": query}]
                    ))
                    
                    # Call the specialist agent
                    try:
                        print(f"Calling specialist agent: {agent_name}")
                        response = await agent.process_request(query, user_id, session_id, agent_history)
                        response_text = self._extract_response_text(response)
                        
                        # Store the response
                        specialist_responses.append({
                            'agent': agent_name,
                            'query': query,
                            'response': response_text
                        })
                        
                        # Update agent history
                        agent_history.append(ConversationMessage(
                            role=ParticipantRole.ASSISTANT,
                            content=[{"text": response_text}]
                        ))
                    except Exception as e:
                        print(f"Error calling agent {agent_name}: {str(e)}")
                        specialist_responses.append({
                            'agent': agent_name,
                            'query': query,
                            'error': str(e)
                        })
                else:
                    print(f"Agent not found: {agent_name}")

                    # Step 3: Handle different result scenarios
            
            elif action_type == "parallel_group":
                # Ensure dependencies are met
                depends_on = action.get('depends_on', [])
                
                # Get the parallel tasks to execute
                parallel_actions = action.get('actions', [])
                
                # Create tasks for parallel execution
                parallel_tasks = []
                
                for parallel_action in parallel_actions:
                    agent_name = parallel_action.get('agent')
                    query = parallel_action.get('query', user_input)
                    output_var = parallel_action.get('output_var')
                    
                    # Process variable substitutions
                    if isinstance(query, str):
                        for var_name in depends_on:
                            if var_name in intermediate_results:
                                query = query.replace(f"{{{{{var_name}}}}}", intermediate_results[var_name])
                    
                    if agent_name in self.agents:
                        # Create task for parallel execution
                        task = self._process_agent_request(
                            agent_name, query, user_id, session_id, output_var
                        )
                        parallel_tasks.append(task)
                
                # Execute tasks in parallel
                parallel_responses = await asyncio.gather(*parallel_tasks, return_exceptions=True)
                
                # Process results
                for response in parallel_responses:
                    if isinstance(response, Exception):
                        print(f"Error in parallel execution: {str(response)}")
                    else:
                        # Add to specialist_responses
                        specialist_responses.append(response['response_data'])
                        
                        # Store output variable if specified
                        if 'output_var' in response and 'response_text' in response:
                            intermediate_results[response['output_var']] = response['response_text']
                        
        # Case 1: Direct response from supervisor
        if direct_response:
            final_response = direct_response
            response_source = "supervisor_direct"

        # Case 2: Single specialist response
        elif len(specialist_responses) == 1 and 'response' in specialist_responses[0]:
            final_response = specialist_responses[0]['response']
            response_source = specialist_responses[0]['agent']

        # Case 3: Multiple specialist responses needing synthesis
        elif len(specialist_responses) > 1:
            # Format specialist responses for synthesis
            specialist_info = "\n\n".join([
                f"[{resp['agent']} RESPONSE TO '{resp['query']}']\n{resp.get('response', 'ERROR: ' + resp.get('error', 'Unknown error'))}" 
                for resp in specialist_responses
            ])
                        
            synthesis_input = f"""TASK: Synthesize specialist responses into a coherent response for the user.

                SPECIALIST RESPONSES:
                {specialist_info}

                INSTRUCTIONS:
                1. Read the specialist responses
                2. Combine the information into a single response
                3. Provide the synthesized response"""
                
            # Send to supervisor
            synthesis_response = await self.supervisor.process_request(
                synthesis_input, user_id, session_id, history
            )
            
            final_response = self._extract_response_text(synthesis_response)
            response_source = "synthesis"
        
        # Case 4: No sucessful responses
        else:
            final_response = "I apologize, but I encountered an issue while processing your request. Could you please try again or rephrase your question?"
            response_source = "error_fallback"

        # Add final response to main conversation history
        history.append(ConversationMessage(
            role=ParticipantRole.ASSISTANT,
            content=[{"text": final_response}]
        ))

        # Create metadata for response
        metadata = {
            "source": response_source,
            "agent_count": len(specialist_responses),
            "plan": plan.get("reasoning", "No reasoning provided")
        }

        # Return the final response
        return AgentResponse(
            output=final_response,
            metadata=metadata,
            streaming=False
        )