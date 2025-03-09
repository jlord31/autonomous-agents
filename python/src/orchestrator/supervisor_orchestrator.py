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
        self.last_active_agent = {}
        
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
        """Extract the execution plan from supervisor response with better error handling"""
        try:
            # Initialize empty plan - will be populated either by JSON parsing or fallback
            plan = {"reasoning": "Extracted from text response", "actions": []}
            json_parsed = False
            
            # Look for JSON between ``` markers
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            
            if not json_match:
                # Try to find standalone JSON
                json_match = re.search(r'(\{\s*"reasoning".*\})', response_text, re.DOTALL)
                
            if json_match:
                plan_json = json_match.group(1)
                
                # Basic JSON error correction - fix common formatting issues
                plan_json = re.sub(r'"\s*\n\s*"', '",\n"', plan_json)  # Add missing commas
                # Add missing comma after object properties
                plan_json = re.sub(r'"([^"]*)"(\s*)\}', r'"\1"\2}', plan_json)
                # Add missing comma between properties 
                plan_json = re.sub(r'"\s*"', '", "', plan_json)
                
                try:
                    parsed_plan = json.loads(plan_json)
                    plan = parsed_plan  # Replace the empty plan with parsed JSON
                    json_parsed = True
                    print("Successfully parsed supervisor JSON response")
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {str(e)}")
                    # Continue with fallback methods
            
            # Only apply name scanning as fallback if JSON parsing failed
            if not json_parsed:
                print("Using fallback agent name detection")
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
        """Helper to extract text from various response types with improved error handling"""
        try:
            if hasattr(response, 'content') and isinstance(response.content, list):
                result = ""
                for content_block in response.content:
                    # Add defensive checking
                    if isinstance(content_block, dict) and "text" in content_block and content_block["text"] is not None:
                        result += content_block["text"]
                return result
            elif hasattr(response, 'output'):
                return response.output if response.output is not None else ""
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            print(f"Error extracting response text: {str(e)}")
            return "Error extracting response"
        
    async def _process_agent_request(self, agent_name, query, user_id, session_id, output_var=None):
        """Process a request to a specialist agent for parallel execution"""
        agent = self.agents[agent_name]
        agent_history = self._get_agent_history(session_id, agent_name)
        
        # Add query to agent history
        agent_history.append(ConversationMessage(
            role=ParticipantRole.USER,
            content=[{"text": query if query is not None else ""}]  # Add null check
        ))
        
        try:
            print(f"Calling specialist agent (parallel): {agent_name}")
            response = await agent.process_request(query, user_id, session_id, agent_history)
            response_text = self._extract_response_text(response)
            
            if response_text:  # Only add if we have text
                agent_history.append(ConversationMessage(
                    role=ParticipantRole.ASSISTANT,
                    content=[{"text": response_text}]
                ))
            
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
        
        # Check if this is a follow-up to a previous agent interaction
        last_agent = self.last_active_agent.get(session_id)
        
        if last_agent and last_agent in self.agents:
            
            # Get the agent's capabilities
            agent_capabilities = ""
            
            agent = self.agents[last_agent]
            if hasattr(agent, 'description'):
                agent_capabilities = agent.description
            
            # Include more context for accurate continuity determination
            agent_history = self._get_agent_history(session_id, last_agent)
            recent_exchanges = ""
            
            # Get the most recent exchange (last 2 messages if available)
            if len(agent_history) >= 2:
                user_msg = agent_history[-2].content[0].get("text", "") if agent_history[-2].content else ""
                agent_msg = agent_history[-1].content[0].get("text", "") if agent_history[-1].content else ""
                recent_exchanges = f"Last user message: {user_msg}\nLast agent response: {agent_msg}"

    
            # This could be a follow-up - ask the supervisor with better context
            continuity_input = f"""TASK: Determine if this user message is a follow-up to the previous conversation with {last_agent}.

                PREVIOUS AGENT: {last_agent}
                AGENT CAPABILITIES: {agent_capabilities}

                RECENT CONVERSATION:
                {recent_exchanges}

                NEW USER REQUEST: {user_input}

                INSTRUCTIONS:
                1. Read the previous agent response and user's new request carefully
                2. Determine if the NEW REQUEST is directly related to what {last_agent} was helping with
                3. Respond with ONLY "YES" if the same agent should continue the conversation
                4. Respond with ONLY "NO" if this is a new topic or request better handled by a different agent
            """
            
            continuity_response = await self.supervisor.process_request(
                continuity_input, user_id, session_id, history[:-1]  
            )
            continuity_text = self._extract_response_text(continuity_response).strip().upper()
                
            if "YES" in continuity_text:
                print(f"Continuing conversation with previous agent: {last_agent}")
                # Direct the request to the previous agent
                agent = self.agents[last_agent]
                agent_history = self._get_agent_history(session_id, last_agent)
                
                # Add user query to agent history
                agent_history.append(ConversationMessage(
                    role=ParticipantRole.USER,
                    content=[{"text": user_input}]
                ))
                
                # Send to the agent
                response = await agent.process_request(user_input, user_id, session_id, agent_history)
                response_text = self._extract_response_text(response)
                
                # Update agent history
                agent_history.append(ConversationMessage(
                    role=ParticipantRole.ASSISTANT,
                    content=[{"text": response_text}]
                ))
                
                # Update main history
                history.append(ConversationMessage(
                    role=ParticipantRole.ASSISTANT,
                    content=[{"text": response_text}]
                ))
                
                # Return the response
                return AgentResponse(
                    output=response_text,
                    metadata={"source": last_agent, "agent_count": 1, "plan": "Direct continuation"},
                    streaming=False
                )
        
        
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
                response_text = action.get('response', '')
                # Process variable substitutions if needed
                for var_name, var_value in intermediate_results.items():
                    placeholder = f"{{{{{var_name}}}}}"
                    if placeholder in response_text:
                        response_text = response_text.replace(placeholder, var_value)
                direct_response = response_text
                continue
                
            # Handle specialist agent calls
            elif action_type == "call_specialist":
                #print(f"DEBUG: Processing query for {agent_name}: {query[:50]}..." if query else "DEBUG: Query is None")
                agent_name = action.get('agent')
                query = action.get('query', user_input)
                output_var = action.get('output_var')
                
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
                        
                        if output_var:
                            intermediate_results[output_var] = response_text
                            print(f"Stored result in variable {output_var}")
            
                        
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
            self.last_active_agent[session_id] = specialist_responses[0].get('agent')

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
            "plan": plan.get("reasoning", "No reasoning provided"),
            #"tools_used": ', '.join(tool_calls)
        }

        # Return the final response
        return AgentResponse(
            output=final_response,
            metadata=metadata,
            streaming=False
        )