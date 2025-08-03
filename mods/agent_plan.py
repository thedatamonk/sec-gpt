from tools import *
from typing import List, Dict, Optional
from schemas.schema import ReACTResponseSchema
from mods.utils import llm_call

class PlanningAgent:

    SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self):
        # Instantiate all available tools
        self.available_tools: List[BaseTool] = [
            SearchFilingsTool(),
            GetFilingContentTool(),
            ExtractDataTool(),
            CalculateTool(),
        ]

        # Create a mapping from tool name to tool instance for easy lookup
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in self.available_tools}

        # Dynamically generate the tool description for the prompt
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.available_tools])
        
        self.system_prompt = \
f"""
You are an expert financial analyst agent. Your goal is to answer the user's query by breaking it down into a series of steps.
At each step, you must reason with a "thought" and then decide on an "action" by calling one of the available tools.

Available Tools:
{tool_descriptions}
- Finish: Provides the final, synthesized answer to the user. Use this when you have all the information. Parameters: answer.

You MUST output your response as a single, valid JSON object. Do NOT output any other text.
"""
    
    def run(self, user_query: str, max_steps: int = 15, history: Optional[List] = None):
        
        print(f"--- Starting Agent with Query: '{user_query}' ---\n")
        scratchpad = f"You must answer the following query: '{user_query}'\n"


        for i in range(max_steps):
            print(f"--- Step {i+1} ---")
            
            # 1. THINK
            prompt = f"{self.system_prompt}\n\nThis is your scratchpad, containing the history of your previous steps:\n{scratchpad}"
            response = llm_call(messages=[{"role": "user", "content": prompt}], response_format=ReACTResponseSchema.model_json_schema())
            response_obj = ReACTResponseSchema.model_validate_json(response).model_dump()

            thought = response_obj["thought"]
            tool_name = response_obj["action"]["tool_name"]
            action_params = response_obj["action"]["parameters"]

            print(f"Thought: {thought}")
            scratchpad += f"\nThought: {thought}"

            # 2. CHECK FOR COMPLETION
            if tool_name == "Finish":
                final_answer = action_params.get('answer', "No answer provided.")
                print(f"\n--- Agent Finished ---\nFinal Answer: {final_answer}")
                return {"final_response": final_answer, "scratchpad": scratchpad}
            
            # 3. ACT
            action_str = f"{tool_name}({action_params})"
            print(f"Action: {action_str}")
            scratchpad += f"\nAction: {action_str}"

            tool_to_execute = self.tools.get(tool_name)

            if not tool_to_execute:
                observation = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    observation = tool_to_execute.execute(**action_params)
                except Exception as e:
                    observation = f"Error executing tool '{tool_name}': {e}"


            # 4. OBSERVE
            print(f"Observation: {observation}\n")
            scratchpad += f"\nObservation: {observation}"


            
        print("\n--- Agent Stopped: Max steps reached. ---")
        return {"final_response": "Couldn't reach FINISH state", "scratchpad": scratchpad}



if __name__ == "__main__":
    agent = PlanningAgent()
    query = "Compare the revenue growth of Apple and Microsoft in 2023. What was the percentage change for each from 2022?"
    agent.run(user_query=query, max_steps=5)