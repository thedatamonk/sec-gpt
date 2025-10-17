import logging
from typing import Any, Dict, List, Optional

from mods.llm import BaseLLM, OpenAILLM
from mods.query_validator import QueryValidator
from sec_edgar_mcp.tools.company import CompanyTools
from sec_edgar_mcp.tools.financial import FinancialTools
from sec_edgar_mcp.tools.filings import FilingsTools

from templates.template import PromptTemplates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecAgent:
    """
    An agent designed to answer queries about SEC filings by following a
    Reason-Act (ReAct) framework.
    """

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.query_validator = QueryValidator()
        
        # Initialize tool instances
        self.company_tools = CompanyTools()
        self.financial_tools = FinancialTools()
        self.filing_tools = FilingsTools()
        
        # Map tool names to (instance, method_name)
        # self.tool_registry = {
        #     "get_cik_by_ticker": (self.company_tools, "get_cik_by_ticker"),
        #     "get_company_info": (self.company_tools, "get_company_info"),
        #     "search_companies": (self.company_tools, "search_companies"),
        #     "get_company_facts": (self.company_tools, "get_company_facts"),
        # }

        # Auto-register all decorated methods from tool classes
        self.tool_registry = {}
        self._register_tool_class(CompanyTools, self.company_tools)
        self._register_tool_class(FinancialTools, self.financial_tools)
        self._register_tool_class(FilingsTools, self.filing_tools)


        # Use provided LLM or default to OpenAI
        self.llm = llm or OpenAILLM(model="gpt-5")

        # Define tools in OpenAI function format
        self.tool_definitions = self._create_tool_definitions()

        logger.info(f"Initialized SecAgent with {len(self.tool_registry)} tools")   
    
    
    def _register_tool_class(self, tool_class, instance):
        """
        Register all decorated methods from a tool class.
        
        Args:
            tool_class: The tool class (e.g., CompanyTools)
            instance: Instance of the tool class
        """
        for method_name in tool_class.get_method_names():
            self.tool_registry[method_name] = (instance, method_name)
            logger.info(f"Registered tool: {method_name}")

    
    def _create_tool_definitions(self) -> List[Dict]:
        """
        Collect tool definitions from all registered tool classes.
        
        Returns:
            List of OpenAI function definitions
        """
        all_definitions = []
        
        # Collect from CompanyTools
        all_definitions.extend(CompanyTools.get_tool_definitions())
        all_definitions.extend(FinancialTools.get_tool_definitions())
        all_definitions.extend(FilingsTools.get_tool_definitions())
        
        # Future: Add more tool classes here
        # all_definitions.extend(FinancialTools.get_tool_definitions())
        # all_definitions.extend(FilingTools.get_tool_definitions())
        
        return all_definitions

    def run(self, query: str) -> str:
        """Process user query and return response"""

        # Stage 1 - Validate query scope and extract entities
        validation_result: Dict[str, Any] = self.query_validator.validate_and_enrich(query)

        # Check if validation_result has an attribute "enriched_query"
        if "enriched_query" in validation_result.keys():
            extracted_entities = validation_result["enriched_query"]
        else:
            return validation_result.get("message", "Invalid query.")
        
        try:
            logger.info(f"Processing query: {query}")
            logger.info(f"Extracted entities: {extracted_entities}")

            # Phase 1: Create execution plan
            plan = self._create_plan(query, extracted_entities)
            if not plan:
                return "I couldn't create a plan to answer your query. Please try rephrasing."
        
            logger.info(f"Generated plan with {len(plan)} steps")

            # Phase 2: Execute plan step by step (Fail Fast)
            step_results = self._execute_plan(plan)

            # Phase 3: Synthesize final answer
            final_answer = self._synthesize_answer(query, plan, step_results)
            
            return final_answer

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"Sorry, I encountered an error processing your query: {str(e)}"
        
    def _create_plan(self, user_query: str, extracted_entities: Dict[str, Any]) -> Optional[List[Dict]]:
        """
        Create execution plan using LLM
        Returns: List of step dictionaries or None if planning failed
        """

        context = self._build_context(user_query, extracted_entities)

        # get tool definitions
        tools_info = self._format_tools_for_prompt()


        planning_prompt = PromptTemplates.PLANNING_TEMPLATE.substitute(context=context, tools_info=tools_info)

        try:
            messages = [{"role": "user", "content": planning_prompt}]
            response = self.llm.call(messages=messages)

            if not response["content"]:
                logger.error("LLM returned empty planning response")
                return None
            
            # Parse JSON plan
            import json
            plan_json = json.loads(response["content"])
            
            if "plan" not in plan_json or not isinstance(plan_json["plan"], list):
                logger.error("Invalid plan structure")
                return None
            
            logger.info(f"Plan created successfully: {json.dumps(plan_json, indent=2)}")
            return plan_json["plan"]
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.error(f"LLM response was: {response['content']}")
            return None
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return None

    def _format_tools_for_prompt(self) -> str:
        """Format tool definitions into human-readable text for planning prompt"""
        tools_text = "Available Tools:\n\n"
        
        for tool_def in self.tool_definitions:
            func = tool_def["function"]
            name = func["name"]
            description = func["description"]
            params = func["parameters"]["properties"]
            required = func["parameters"].get("required", [])
            
            tools_text += f"Tool: {name}\n"
            tools_text += f"Description: {description}\n"
            tools_text += "Parameters:\n"
            
            for param_name, param_info in params.items():
                param_type = param_info["type"]
                param_desc = param_info.get("description", "")
                is_required = "REQUIRED" if param_name in required else "optional"
                
                tools_text += f"  - {param_name} ({param_type}, {is_required}): {param_desc}\n"
                
                # Add enum values if present
                if "enum" in param_info:
                    tools_text += f"    Valid values: {', '.join(map(str, param_info['enum']))}\n"
            
            tools_text += "\n"
        
        return tools_text

    def _execute_plan(self, plan: List[Dict]) -> List[Dict]:
        """
        Execute plan step by step with fail-fast behavior
        Returns: List of step results
        Raises: Exception if any critical step fails
        """
        step_results = []
        
        for step in plan:
            step_num = step.get('step', '?')
            action_type = step.get('action_type')
            description = step.get('description', 'Unknown step')
            
            logger.info(f"Executing Step {step_num}: {description} (type: {action_type})")
            
            if action_type == 'reasoning':
                # Reasoning steps are completed during planning
                result = {
                    'step': step_num,
                    'description': description,
                    'status': 'completed',
                    'action_type': 'reasoning',
                    'output': step.get('expected_output'),
                    'reasoning': step.get('reasoning')
                }
                logger.info(f"Step {step_num} (reasoning): Completed")
            
            elif action_type == 'tool_call':
                tool_name = step.get('tool')
                tool_params = step.get('tool_parameters', {})
                
                if not tool_name or tool_name not in self.tool_registry:
                    error_msg = f"Step {step_num} failed: Invalid tool '{tool_name}'"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                # Get tool instance and method name
                tool_instance, method_name = self.tool_registry[tool_name]
                
                # Call the tool method directly
                try:
                    tool_method = getattr(tool_instance, method_name)
                    tool_result = tool_method(**tool_params)
                except AttributeError:
                    error_msg = f"Step {step_num} failed: Method '{method_name}' not found on tool"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                except TypeError as e:
                    error_msg = f"Step {step_num} failed: Invalid parameters for {tool_name}: {str(e)}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                # Check if tool execution was successful
                if not tool_result.get("success", False):
                    # FAIL FAST - Stop execution on tool failure
                    error_msg = f"Step {step_num} failed: {tool_result.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                result = {
                    'step': step_num,
                    'description': description,
                    'status': 'success',
                    'action_type': 'tool_call',
                    'tool': tool_name,
                    'parameters': tool_params,
                    'output': tool_result,
                    # 'metadata': tool_result.metadata,
                    'expected_output': step.get('expected_output')
                }
                logger.info(f"Step {step_num} (tool_call): Success - {tool_name}")

            elif action_type == 'synthesis':
                # Synthesis will be handled separately after all steps
                result = {
                    'step': step_num,
                    'description': description,
                    'status': 'pending',
                    'action_type': 'synthesis',
                    'expected_output': step.get('expected_output')
                }
                logger.info(f"Step {step_num} (synthesis): Pending")
            
            else:
                error_msg = f"Step {step_num} failed: Unknown action_type '{action_type}'"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            step_results.append(result)

        logger.info(f"All {len(step_results)} steps executed successfully")
        return step_results

    def _synthesize_answer(self, user_query: str, plan: List[Dict], 
                          step_results: List[Dict]) -> str:
        """
        Synthesize final answer from execution results
        """
        # Build context for synthesis
        synthesis_context = \
f"""Original Question: {user_query}

Execution Plan and Results:
"""
        for result in step_results:
            step_num = result['step']
            desc = result['description']
            action_type = result['action_type']
            status = result['status']
            
            synthesis_context += f"\nStep {step_num}: {desc} [{action_type}]\n"
            synthesis_context += f"Status: {status}\n"
            
            if action_type == 'reasoning':
                synthesis_context += f"Output: {result.get('output')}\n"
            elif action_type == 'tool_call':
                synthesis_context += f"Tool Used: {result.get('tool')}\n"
                synthesis_context += f"Result: {result.get('output')}\n"
        
        synthesis_context += "\nBased on the execution results above, provide a clear, natural language answer to the user's question. Be concise and accurate."
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful SEC financial data assistant. Synthesize the execution results into a clear, natural language answer."
                },
                {
                    "role": "user",
                    "content": synthesis_context
                }
            ]
            
            response = self.llm.call(messages=messages)
            final_answer = response.get("content", "I processed your query but couldn't generate a response.")
            
            logger.info("Final answer synthesized successfully")
            return final_answer

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"I retrieved the data but encountered an error formatting the response: {str(e)}"
    
    def _build_context(self, user_query: str, extracted_entities: Dict) -> str:
        """Build enriched context for LLM from query and extracted entities"""
        context = f"User Query: {user_query}\n\n"
        context += "Extracted Information:\n"

        # Add company info
        if extracted_entities.get("companies"):
            companies = extracted_entities["companies"]
            context += f"- Companies: {', '.join([c['value'] for c in companies])}\n"
        
        # Add financial metrics
        if extracted_entities.get("financial_metrics"):
            metrics = extracted_entities["financial_metrics"]
            metric_names = [str(m).split('.')[-1] if hasattr(m, 'value') else str(m) 
                          for m in metrics]
            context += f"- Financial Metrics: {', '.join(metric_names)}\n"
        
        # Add time period
        if extracted_entities.get("time_period"):
            periods = extracted_entities["time_period"]
            context += f"- Time Period: {', '.join(periods)}\n"

        context += "\nPlease use the available tools to answer this query accurately."
        return context


if __name__ == "__main__":

    # ANSI escape codes for colors
    USER_COLOR = "\033[94m"      # Blue
    AGENT_COLOR = "\033[92m"     # Green
    RESET_COLOR = "\033[0m"

    # Instantiate the agent once
    sec_agent = SecAgent()

    queries_to_test = ["Show me AAPL 10-K document from Q1 2024.", "Show me Nvidia's info. Include ticker name and CIK."]

    response = sec_agent.run(queries_to_test[0])

    print (response)
    

    # print(f"{AGENT_COLOR}{'='*50}{RESET_COLOR}")
    # print (f"{AGENT_COLOR}Start chatting with SEC-Agent!!\nType \\quit to end the conversation.{RESET_COLOR}")
    # print(f"{AGENT_COLOR}{'='*50}{RESET_COLOR}")
    
    # while True:
    #     user_query = input(f"{USER_COLOR}You: {RESET_COLOR}")

    #     if user_query.strip().lower() == "\\quit":
    #         print(f"{AGENT_COLOR}SEC-Agent:{RESET_COLOR} Bye!")
    #         break
        
    #     # Run the agent with the current query and the full history
    #     # Here user starts a new chat session
    #     agent_response = sec_agent.run(user_query)
        
    #     print (f"{AGENT_COLOR}SEC-Agent:{RESET_COLOR} {agent_response}'")
