import openlit
import json
from typing import List, Dict, Any, TypedDict, Optional, Callable
import ollama
from mods.query_validator import QueryValidator
from mods.agent_plan import PlanningAgent
from mods.utils import llm_call



# AgentState
class AgentState(TypedDict):
    original_query: str
    conversation_history: List[Dict[str, str]]
    is_valid_query: bool
    requires_clarification: bool
    clarification_question: Optional[str]
    execution_plan: List[Dict[str, Any]]
    evidence: Dict[int, Any] # TODO: Keyed by step_id -> What is step_id?
    final_answer: Optional[str]
    error_message: Optional[str]


class SecAgent:
    """
    An agent designed to answer queries about SEC filings by following a
    Reason-Act (ReAct) framework.
    """

    def __init__(self):
        self.state: AgentState = self._initialize_state()
        self.query_validator = QueryValidator()
        self.planner = PlanningAgent()
    
    def _initialize_state(self, query: str = "", history: Optional[List] = None) -> AgentState:
        """
        TODO: I still don't know why do we need this. WTF are we initialising?
        Creates a fresh state for a new query.
        """

        # Resetting history when user starts a new chat session
        if history is None:
            history = []
        
        # We are returning the initial state of the agent
        return {
            "original_query": query,
            "conversation_history": history,
            "is_valid_query": False,
            "requires_clarification": False,
            "clarification_question": None,
            "execution_plan": [],
            "evidence": {},
            "final_answer": None,
            "error_message": None
        }
    
#     def _synthesize_final_answer(self):
#             """Uses an LLM call to synthesize the final answer from the evidence."""
#             print("\n[Phase 4: Synthesizing Final Answer]")
#             prompt = \
# f"""
# You are a financial assistant. Your task is to answer the user's original query based *ONLY* on the evidence provided below.
# Do not use any of your internal knowledge. Do not make assumptions.
# For every factual claim or number you state, you MUST cite the source.

# User's Original Query: '{self.state['original_query']}'
# Conversation History: {self.state['conversation_history']}

# Evidence Collected:
# ---
# {json.dumps(self.state['evidence'], indent=2)}
# ---

# Provide a comprehensive final answer below.
# """
#             self.state['final_answer'] = llm_call(prompt)

    def run(self, user_query: str, conversation_history: Optional[List] = None) -> Dict[str, Any]:
        """The main entry point for processing a user's query."""

        # Initialise the agent state
        # TODO: Still have to think do we really need this?
        self.state = self._initialize_state(user_query, conversation_history)

        # Query validation and expansion and clarification question should be generated as part of this response
        # NOTE: Currently history is not being used anywhere in query validation
        # But ideally it should be
        response = self.query_validator.validate_and_enrich(user_query, history=conversation_history)

        if response["status"] == "valid":
            final_answer = self.planner.run(user_query=response["enriched_query"], max_steps=5, history=conversation_history)
        else:
            return response
        
        return {"final_answer": final_answer}

if __name__ == "__main__":

    # ANSI escape codes for colors
    USER_COLOR = "\033[94m"      # Blue
    AGENT_COLOR = "\033[92m"     # Green
    RESET_COLOR = "\033[0m"

    # Instantiate the agent once
    sec_agent = SecAgent()
    
    # Maintain history across a conversation
    conversation_history = []

    print(f"{AGENT_COLOR}{'='*50}{RESET_COLOR}")
    print (f"{AGENT_COLOR}Start chatting with SEC-Agent!!\nType \\quit to end the conversation.{RESET_COLOR}")
    print(f"{AGENT_COLOR}{'='*50}{RESET_COLOR}")
    
    while True:
        user_query = input(f"{USER_COLOR}You: {RESET_COLOR}")

        if user_query.strip().lower() == "\\quit":
            print(f"{AGENT_COLOR}SEC-Agent:{RESET_COLOR} Bye!")
            break
        
        # Run the agent with the current query and the full history
        # Here user starts a new chat session
        agent_response = sec_agent.run(user_query, conversation_history)
        
        print (f"{AGENT_COLOR}SEC-Agent:{RESET_COLOR} {agent_response}'")

        # Update the history with the latest turn
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "agent", "content": agent_response})
