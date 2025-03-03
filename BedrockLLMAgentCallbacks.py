from multi_agent_orchestrator.agents import ( BedrockLLMAgent, BedrockLLMAgentOptions, AgentResponse, AgentCallbacks )

class BedrockLLMAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        # handle response streaming here
        print(token, end='', flush=True)