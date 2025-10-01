import logging
from typing import Any, Dict, List, Optional

from mods.llm import BaseLLM, OpenAILLM
from mods.query_validator import QueryValidator
from mods.tools import CompanyLookupTool, FilingSearchTool, FinancialDataTool

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
        self.tools = {
            "company_lookup": CompanyLookupTool(),
            "financial_data": FinancialDataTool(),
            "filing_search": FilingSearchTool()
        }

        # Use provided LLM or default to OpenAI
        self.llm = llm or OpenAILLM(model="gpt-3.5-turbo")

        # Define tools in OpenAI function format
        self.tool_definitions = self._create_tool_definitions()
    
    
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
            # Build context for LLM
            context = self._build_context(query, extracted_entities)

            # Initial LLM call with tools
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful SEC financial data assistant. Use the provided tools to answer user questions about companies and their financial data. Always call appropriate tools to get accurate information."
                },
                {
                    "role": "user", 
                    "content": context
                }
            ]

            # Call LLM with tool definitions
            # Here we are calling the LLM to respond with the list of tools that would be required to answer the query
            llm_response = self.llm.call(
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto"
            )

            # Execute any tool calls requested by LLM
            if llm_response["tool_calls"]:
                tool_results = self._execute_tool_calls(llm_response["tool_calls"])

                # Add assistant message with tool calls (use raw format for OpenAI)
                assistant_msg = {
                    "role": "assistant",
                    "content": llm_response["content"]
                }
                if llm_response.get("raw_tool_calls"):
                    assistant_msg["tool_calls"] = llm_response["raw_tool_calls"]
                
                messages.append(assistant_msg)

                # Add tool results as separate messages
                for tool_call, result in zip(llm_response["tool_calls"], tool_results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "content": str(result)
                    })

                # Get final response from LLM
                final_response = self.llm.call(messages=messages)
                return final_response["content"] or "I processed your query but couldn't generate a response."
            
            # If no tool calls, return direct response
            return llm_response["content"] or "I couldn't determine how to answer your query."

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"Sorry, I encountered an error processing your query: {str(e)}"

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

    def _create_tool_definitions(self) -> List[Dict]:
        """Create OpenAI function definitions for our tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "company_lookup",
                    "description": "Look up company information by ticker or CIK",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cik_or_ticker": {
                                "type": "string",
                                "description": "Company ticker (e.g., AAPL) or CIK number"
                            }
                        },
                        "required": ["cik_or_ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "financial_data",
                    "description": "Get financial metrics (revenue, earnings, cash flow) for a specific period",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cik_or_ticker": {
                                "type": "string",
                                "description": "Company ticker (e.g., AAPL) or CIK number"
                            },
                            "year": {
                                "type": "integer",
                                "description": "Year for the financial data (e.g., 2020)"
                            },
                            "quarter": {
                                "type": "integer",
                                "description": "Quarter (1-4) for quarterly data, omit for annual data",
                                "enum": [1, 2, 3, 4]
                            },
                            "metric": {
                                "type": "string",
                                "description": "Financial metric to retrieve",
                                "enum": ["revenue", "net_income", "total_assets", "total_equity", "cash_flow_from_operating_activities"]
                            }
                        },
                        "required": ["cik_or_ticker", "year", "metric"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "filing_search",
                    "description": "Search for SEC filings by company and form type",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cik_or_ticker": {
                                "type": "string",
                                "description": "Company ticker (e.g., AAPL) or CIK number"
                            },
                            "form_type": {
                                "type": "string",
                                "description": "Type of SEC filing",
                                "enum": ["10-K", "10-Q"]
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of filings to return (default 5)",
                                "default": 5
                            },
                            "year": {
                                "type": "integer",
                                "description": "Filter by specific year"
                            }
                        },
                        "required": ["cik_or_ticker"]
                    }
                }
            }
        ]

    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[str]:
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")

            if tool_name not in self.tools:
                # TODO: If tool not foundt, then we proceed with other tools
                # But is this the correct behavior?
                results.append(f"Error: Tool '{tool_name}' not found")
                continue

            try:
                tool = self.tools[tool_name]
                result = tool.execute(**arguments)
                
                # Format result for LLM
                if result.success:
                    result_str = f"Success: {result.data}"
                    if result.metadata:
                        result_str += f"\nMetadata: {result.metadata}"
                else:
                    result_str = f"Error: {result.error}"
                
                results.append(result_str)
                
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                results.append(f"Error executing {tool_name}: {str(e)}")
        
        return results


if __name__ == "__main__":

    # ANSI escape codes for colors
    USER_COLOR = "\033[94m"      # Blue
    AGENT_COLOR = "\033[92m"     # Green
    RESET_COLOR = "\033[0m"

    # Instantiate the agent once
    sec_agent = SecAgent()

    queries_to_test = ["What's AAPL's profit for last year?", ""]

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
