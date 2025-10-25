import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from mods.llm import BaseLLM, OpenAILLM
from mods.query_validator import QueryValidator
from sec_edgar_mcp.tools.company import CompanyTools
from sec_edgar_mcp.tools.filings import FilingsTools
from sec_edgar_mcp.tools.financial import FinancialTools
from templates.template import PromptTemplates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types for replanning strategy"""
    RECOVERABLE = "recoverable"      # Can try alternatives
    RATE_LIMIT = "rate_limit"        # Need to wait and retry
    UNRECOVERABLE = "unrecoverable"  # Can't fix, fail gracefully


FALLBACK_STRATEGIES = {
    # Filing not found - try different form types or periods
    "filing_not_found": [
        {
            "description": "Try alternative filing form type",
            "alternative_forms": {"10-K": "10-Q", "10-Q": "10-K"}
        },
        {
            "description": "Try previous year's filing",
            "action": "adjust_year",
            "offset": -1
        }
    ],
    
    # Company not found - try fuzzy matching
    "company_not_found": [
        {
            "description": "Search for company by name instead",
            "action": "use_search_tool"
        },
        {
            "description": "Try removing common suffixes",
            "remove_suffixes": [" Inc", " Corp", " LLC", " Corporation"]
        }
    ],
    
    # Metric/data not found - try alternative names or approaches
    "data_not_found": [
        {
            "description": "Try alternative metric names",
            "metric_alternatives": {
                "revenue": ["Revenues", "RevenueFromContractWithCustomer", "SalesRevenueNet"],
                "net_income": ["NetIncomeLoss", "ProfitLoss"],
                "cash_flow": ["NetCashProvidedByUsedInOperatingActivities"]
            }
        },
        {
            "description": "Use company facts API instead",
            "action": "use_facts_api"
        }
    ]
}
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

        # Auto-register all decorated methods from tool classes
        self.tool_registry = {}
        self._register_tool_class(CompanyTools, self.company_tools)
        self._register_tool_class(FinancialTools, self.financial_tools)
        self._register_tool_class(FilingsTools, self.filing_tools)


        # Use provided LLM or default to OpenAI
        self.llm = llm or OpenAILLM(model="gpt-5")

        # Define tools in OpenAI function format
        self.tool_definitions = self._create_tool_definitions()

        # Replanning configuration
        self.max_replanning_attempts = 2  # Max times to replan per step
        self.max_total_replannings = 5  # Max total replannings per query

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
        
        return all_definitions

    def run(self, query: str) -> Dict[str, Any]:
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
                return {"role": "SECAgent", "content": "I couldn't create a plan to answer your query. Please try rephrasing."}
        
            logger.info(f"Generated plan with {len(plan)} steps")

            # Phase 2: Execute plan step by step (Fail Fast)
            # step_results = self._execute_plan(plan)
            step_results = self._execute_plan_with_replanning(plan, query, extracted_entities)

            # Phase 3: Synthesize final answer
            final_answer = self._synthesize_answer(query, plan, step_results)
            
            return {"role": "SECAgent", "content": final_answer}

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {"role": "SECAgent", "content": f"Sorry, I encountered an error processing your query: {str(e)}"}
        
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

    def _execute_plan_with_replanning(self, plan: List[Dict], original_query: str, extracted_entities: Dict) -> List[Dict]:
        """
        Execute plan with adaptive replanning on failures.
        """
        step_results = []
        total_replannings = 0  # Track total replannings across all steps
        

        for step_index, step in enumerate(plan):
            step_num = step.get('step', '?')
            action_type = step.get('action_type')
            description = step.get('description', 'Unknown step')
            
            logger.info(f"Executing Step {step_num}: {description} (type: {action_type})")

            # Handle reasoning steps
            if action_type == 'reasoning':
                result = {
                    'step': step_num,
                    'description': description,
                    'status': 'completed',
                    'action_type': 'reasoning',
                    'output': step.get('expected_output'),
                    'reasoning': step.get('reasoning')
                }
                logger.info(f"Step {step_num} (reasoning): Completed")
                step_results.append(result)
                continue

            # Handle synthesis steps
            if action_type == 'synthesis':
                result = {
                    'step': step_num,
                    'description': description,
                    'status': 'pending',
                    'action_type': 'synthesis',
                    'expected_output': step.get('expected_output')
                }
                logger.info(f"Step {step_num} (synthesis): Pending")
                step_results.append(result)
                continue

            if action_type == 'tool_call':
                # Track attempted approaches for this step
                attempted_approaches = []
                replanning_attempts = 0
                current_step = step

                while replanning_attempts <= self.max_replanning_attempts:
                    try:
                        # Attempt to execute the current step
                        result = self._execute_single_tool_step(current_step, step_num, description)
                        
                        # Success! Add to results and move on
                        step_results.append(result)
                        logger.info(f"Step {step_num} (tool_call): Success - {current_step.get('tool')}")
                        break
                    
                    except Exception as e:
                        error_msg = str(e)
                        logger.warning(f"Step {step_num} attempt {replanning_attempts + 1} failed: {error_msg}")
                        
                        # Track this attempt
                        attempted_approaches.append({
                            'tool': current_step.get('tool'),
                            'parameters': current_step.get('tool_parameters'),
                            'error': error_msg
                        })

                        # Check if we've exhausted replanning attempts
                        if replanning_attempts >= self.max_replanning_attempts:
                            logger.error(f"Step {step_num}: Exhausted all replanning attempts")
                            # Return partial success with what we have
                            return self._handle_partial_success(
                                completed_steps=step_results,
                                failed_step=current_step,
                                failed_step_num=step_num,
                                remaining_steps=plan[step_index + 1:],
                                error=error_msg
                            )
                        
                        # Check if we've exceeded total replanning budget
                        if total_replannings >= self.max_total_replannings:
                            logger.error(f"Exceeded total replanning budget ({self.max_total_replannings})")
                            return self._handle_partial_success(
                                completed_steps=step_results,
                                failed_step=current_step,
                                failed_step_num=step_num,
                                remaining_steps=plan[step_index + 1:],
                                error="Exceeded maximum replanning attempts for this query"
                            )

                        # Classify the error
                        error_type = self._classify_error(error_msg)
                        logger.info(f"Error classified as: {error_type.value}")

                        # Try to recover based on error type
                        if error_type == ErrorType.UNRECOVERABLE:
                            logger.error(f"Step {step_num}: Unrecoverable error")
                            return self._handle_partial_success(
                                completed_steps=step_results,
                                failed_step=current_step,
                                failed_step_num=step_num,
                                remaining_steps=plan[step_index + 1:],
                                error=error_msg
                            )
                        
                        # Try predefined fallbacks first
                        logger.info(f"Attempting predefined fallback for step {step_num}")
                        fallback_step = self._try_predefined_fallback(
                            current_step, 
                            error_msg, 
                            attempted_approaches
                        )

                        if fallback_step:
                            logger.info("Found predefined fallback, retrying...")
                            current_step = fallback_step
                            replanning_attempts += 1
                            total_replannings += 1
                            continue

                        # No predefined fallback, use LLM replanning
                        logger.info("No predefined fallback, using LLM replanning...")
                        replanned_step = self._replan_with_llm(
                            original_query=original_query,
                            extracted_entities=extracted_entities,
                            failed_step=current_step,
                            error_message=error_msg,
                            completed_steps=step_results,
                            attempted_approaches=attempted_approaches
                        )

                        if replanned_step:
                            logger.info("LLM generated new approach, retrying...")
                            current_step = replanned_step
                            replanning_attempts += 1
                            total_replannings += 1
                        else:
                            logger.error(f"LLM replanning failed, giving up on step {step_num}")
                            return self._handle_partial_success(
                                completed_steps=step_results,
                                failed_step=current_step,
                                failed_step_num=step_num,
                                remaining_steps=plan[step_index + 1:],
                                error="Could not find alternative approach"
                            )
            else:
                # Unknown action type
                error_msg = f"Step {step_num} failed: Unknown action_type '{action_type}'"
                logger.error(error_msg)
                return self._handle_partial_success(
                    completed_steps=step_results,
                    failed_step=step,
                    failed_step_num=step_num,
                    remaining_steps=plan[step_index + 1:],
                    error=error_msg
                )

        logger.info(f"All {len(step_results)} steps executed successfully (with {total_replannings} replannings)")
        return step_results

    def _execute_single_tool_step(
        self, 
        step: Dict, 
        step_num: int, 
        description: str
    ) -> Dict:
        """
        ADDED: Execute a single tool step (extracted from original _execute_plan).
        Raises exception on failure for replanning to catch.
        """
        tool_name = step.get('tool')
        tool_params = step.get('tool_parameters', {})
        
        if not tool_name or tool_name not in self.tool_registry:
            raise Exception(f"Invalid tool '{tool_name}'")
        
        # Get tool instance and method name
        tool_instance, method_name = self.tool_registry[tool_name]
        
        # Call the tool method directly
        try:
            tool_method = getattr(tool_instance, method_name)
            tool_result = tool_method(**tool_params)
        except AttributeError:
            raise Exception(f"Method '{method_name}' not found on tool")
        except TypeError as e:
            raise Exception(f"Invalid parameters for {tool_name}: {str(e)}")
        
        # Check if tool execution was successful
        if not tool_result.get("success", False):
            # Extract error message from tool result
            error_detail = tool_result.get('error', 'Unknown error')
            raise Exception(error_detail)
        
        # Success - return result
        return {
            'step': step_num,
            'description': description,
            'status': 'success',
            'action_type': 'tool_call',
            'tool': tool_name,
            'parameters': tool_params,
            'output': tool_result,
            'expected_output': step.get('expected_output')
        }
    
    def _classify_error(self, error_message: str) -> ErrorType:
        """
        ADDED: Classify error type to determine recovery strategy.
        """
        error_lower = error_message.lower()
        
        # Check for unrecoverable errors
        unrecoverable_patterns = [
            'invalid cik format',
            'authentication failed',
            'network unreachable',
            'invalid tool',
            'method not found'
        ]
        
        for pattern in unrecoverable_patterns:
            if pattern in error_lower:
                return ErrorType.UNRECOVERABLE
        
        # Check for rate limit errors
        rate_limit_patterns = [
            'rate limit',
            '429',
            'too many requests',
            'quota exceeded'
        ]
        
        for pattern in rate_limit_patterns:
            if pattern in error_lower:
                return ErrorType.RATE_LIMIT
        
        # Default to recoverable
        return ErrorType.RECOVERABLE
    
    def _try_predefined_fallback(
        self, 
        failed_step: Dict, 
        error_message: str,
        attempted_approaches: List[Dict]
    ) -> Optional[Dict]:
        """
        ADDED: Try predefined fallback strategies based on error patterns.
        Returns a new step dict if fallback found, None otherwise.
        """
        error_lower = error_message.lower()
        
        # Detect error category
        if 'not found' in error_lower or 'no filings found' in error_lower:
            if 'filing' in error_lower or '10-k' in error_lower or '10-q' in error_lower:
                return self._fallback_filing_not_found(failed_step, attempted_approaches)
            elif 'company' in error_lower or 'cik' in error_lower:
                return self._fallback_company_not_found(failed_step, attempted_approaches)
            else:
                return self._fallback_data_not_found(failed_step, attempted_approaches)
        
        return None
    
    def _fallback_filing_not_found(
        self, 
        failed_step: Dict, 
        attempted_approaches: List[Dict]
    ) -> Optional[Dict]:
        """ADDED: Fallback for filing not found errors."""
        strategies = FALLBACK_STRATEGIES.get("filing_not_found", [])
        
        for strategy in strategies:
            # Try alternative form type
            if "alternative_forms" in strategy:
                original_params = failed_step.get('tool_parameters', {})
                form_type = original_params.get('form_type')
                
                if form_type and form_type in strategy["alternative_forms"]:
                    alt_form = strategy["alternative_forms"][form_type]
                    
                    # Check if we already tried this
                    already_tried = any(
                        a.get('parameters', {}).get('form_type') == alt_form 
                        for a in attempted_approaches
                    )
                    
                    if not already_tried:
                        logger.info(f"Fallback: Trying alternative form {alt_form} instead of {form_type}")
                        new_step = failed_step.copy()
                        new_step['tool_parameters'] = original_params.copy()
                        new_step['tool_parameters']['form_type'] = alt_form
                        return new_step
            
            # Try previous year
            if strategy.get("action") == "adjust_year":
                original_params = failed_step.get('tool_parameters', {})
                year = original_params.get('year')
                
                if year:
                    new_year = year + strategy.get("offset", -1)
                    
                    # Check if we already tried this year
                    already_tried = any(
                        a.get('parameters', {}).get('year') == new_year 
                        for a in attempted_approaches
                    )
                    
                    if not already_tried and new_year >= 2010:  # Don't go too far back
                        logger.info(f"Fallback: Trying year {new_year} instead of {year}")
                        new_step = failed_step.copy()
                        new_step['tool_parameters'] = original_params.copy()
                        new_step['tool_parameters']['year'] = new_year
                        return new_step
        
        return None
    
    def _fallback_company_not_found(
        self, 
        failed_step: Dict, 
        attempted_approaches: List[Dict]
    ) -> Optional[Dict]:
        """ADDED: Fallback for company not found errors."""
        strategies = FALLBACK_STRATEGIES.get("company_not_found", [])
        original_params = failed_step.get('tool_parameters', {})
        identifier = original_params.get('identifier') or original_params.get('cik_or_ticker')
        
        if not identifier:
            return None
        
        for strategy in strategies:
            # Try search tool instead
            if strategy.get("action") == "use_search_tool":
                already_tried_search = any(
                    a.get('tool') == 'search_companies' 
                    for a in attempted_approaches
                )
                
                if not already_tried_search:
                    logger.info(f"Fallback: Using search_companies for '{identifier}'")
                    return {
                        'step': failed_step.get('step'),
                        'description': f"Search for company: {identifier}",
                        'action_type': 'tool_call',
                        'tool': 'search_companies',
                        'tool_parameters': {
                            'query': identifier,
                            'limit': 5
                        },
                        'expected_output': f"Company search results for {identifier}"
                    }
            
            # Try removing suffixes
            if "remove_suffixes" in strategy:
                for suffix in strategy["remove_suffixes"]:
                    if identifier.endswith(suffix):
                        clean_identifier = identifier[:-len(suffix)].strip()
                        
                        already_tried = any(
                            a.get('parameters', {}).get('identifier') == clean_identifier
                            for a in attempted_approaches
                        )
                        
                        if not already_tried:
                            logger.info(f"Fallback: Trying '{clean_identifier}' (removed '{suffix}')")
                            new_step = failed_step.copy()
                            new_step['tool_parameters'] = original_params.copy()
                            new_step['tool_parameters']['identifier'] = clean_identifier
                            return new_step
        
        return None
    
    def _fallback_data_not_found(
        self, 
        failed_step: Dict, 
        attempted_approaches: List[Dict]
    ) -> Optional[Dict]:
        """ADDED: Fallback for data/metric not found errors."""
        strategies = FALLBACK_STRATEGIES.get("data_not_found", [])
        
        for strategy in strategies:
            # Try company facts API
            if strategy.get("action") == "use_facts_api":
                already_tried = any(
                    a.get('tool') == 'get_company_facts' 
                    for a in attempted_approaches
                )
                
                if not already_tried:
                    original_params = failed_step.get('tool_parameters', {})
                    identifier = original_params.get('identifier') or original_params.get('cik_or_ticker')
                    
                    if identifier:
                        logger.info("Fallback: Using get_company_facts API")
                        return {
                            'step': failed_step.get('step'),
                            'description': "Get company facts as alternative",
                            'action_type': 'tool_call',
                            'tool': 'get_company_facts',
                            'tool_parameters': {'identifier': identifier},
                            'expected_output': "Company financial facts"
                        }
        
        return None

    def _replan_with_llm(
        self,
        original_query: str,
        extracted_entities: Dict,
        failed_step: Dict,
        error_message: str,
        completed_steps: List[Dict],
        attempted_approaches: List[Dict]
    ) -> Optional[Dict]:
        """
        ADDED: Use LLM to generate alternative approach when predefined fallbacks fail.
        """
        replanning_prompt = f"""You are helping an SEC filing analysis agent recover from a failed step.

Original User Query: {original_query}

Extracted Entities: {json.dumps(extracted_entities, indent=2)}

Steps Completed Successfully:
{json.dumps([{
    'step': s['step'], 
    'description': s['description'],
    'tool': s.get('tool'),
    'output_summary': 'success' if s['status'] == 'success' else s.get('reasoning', 'pending')
} for s in completed_steps], indent=2)}

Failed Step:
{json.dumps(failed_step, indent=2)}

Error: {error_message}

Attempted Approaches (all failed):
{json.dumps(attempted_approaches, indent=2)}

Available Tools:
{self._format_tools_for_prompt()}

Task: Generate an ALTERNATIVE approach to achieve the goal of the failed step.

Requirements:
1. Use DIFFERENT tools or parameters than what's been tried
2. Consider using data from completed steps if helpful
3. If exact data is unavailable, suggest the closest alternative
4. Return ONLY a single JSON step object with this structure:
{{
  "step": {failed_step.get('step')},
  "description": "New approach description",
  "action_type": "tool_call",
  "tool": "tool_name",
  "tool_parameters": {{"param": "value"}},
  "expected_output": "What this should return",
  "reasoning": "Why this alternative might work"
}}

Return ONLY the JSON object, no other text."""

        try:
            messages = [{"role": "user", "content": replanning_prompt}]
            response = self.llm.call(messages=messages)
            
            if not response.get("content"):
                logger.error("LLM returned empty replanning response")
                return None
            
            # Parse the replanned step
            replanned_step = json.loads(response["content"])
            
            # Validate it has required fields
            if not all(k in replanned_step for k in ['tool', 'tool_parameters']):
                logger.error("Replanned step missing required fields")
                return None
            
            logger.info(f"LLM replanning successful: {replanned_step.get('reasoning', 'No reasoning provided')}")
            return replanned_step
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM replanning response: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM replanning failed: {e}")
            return None

    def _handle_partial_success(
        self,
        completed_steps: List[Dict],
        failed_step: Dict,
        failed_step_num: int,
        remaining_steps: List[Dict],
        error: str
    ) -> List[Dict]:
        """
        ADDED: Handle partial success by returning what was completed.
        Adds special marker for synthesis to explain partial results.
        """
        logger.info(f"Handling partial success: {len(completed_steps)} completed, step {failed_step_num} failed")
        
        # Add a special step indicating partial failure
        partial_result = {
            'step': failed_step_num,
            'description': failed_step.get('description', 'Failed step'),
            'status': 'failed',
            'action_type': 'tool_call',
            'tool': failed_step.get('tool'),
            'parameters': failed_step.get('tool_parameters'),
            'output': None,
            'error': error,
            'partial_success': True
        }
        
        completed_steps.append(partial_result)
        
        # Add note about skipped steps if any
        if remaining_steps:
            skipped_note = {
                'step': 'skipped',
                'description': f"Skipped {len(remaining_steps)} remaining steps due to failure",
                'status': 'skipped',
                'action_type': 'note',
                'skipped_steps': [s.get('description') for s in remaining_steps]
            }
            completed_steps.append(skipped_note)
        
        return completed_steps

    def _synthesize_answer_deprecated(self, user_query: str, plan: List[Dict], 
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
    

    def _synthesize_answer(self, user_query: str, plan: List[Dict], 
                            step_results: List[Dict]) -> str:
        """
        MODIFIED: Synthesize final answer from execution results.
        Now handles partial success cases.
        """
        # Check if this is a partial success
        has_failures = any(r.get('status') == 'failed' for r in step_results)
        
        # Build context for synthesis
        synthesis_context = f"""Original Question: {user_query}

Execution Plan and Results:
"""
        for result in step_results:
                step_num = result['step']
                desc = result['description']
                action_type = result.get('action_type', 'unknown')
                status = result['status']
                
                synthesis_context += f"\nStep {step_num}: {desc} [{action_type}]\n"
                synthesis_context += f"Status: {status}\n"
                
                if action_type == 'reasoning':
                    synthesis_context += f"Output: {result.get('output')}\n"
                elif action_type == 'tool_call':
                    if status == 'success':
                        synthesis_context += f"Tool Used: {result.get('tool')}\n"
                        synthesis_context += f"Result: {result.get('output')}\n"
                    elif status == 'failed':
                        synthesis_context += f"Tool Attempted: {result.get('tool')}\n"
                        synthesis_context += f"Error: {result.get('error')}\n"
                elif action_type == 'note':
                    synthesis_context += f"Note: {result.get('description')}\n"
        
        # ADDED: Special instructions for partial success
        if has_failures:
            synthesis_context += """\n
IMPORTANT: This query resulted in PARTIAL SUCCESS. Some steps failed.

Instructions for your response:
1. Present the information from successful steps clearly
2. Explain what could not be retrieved and why
3. Suggest what the user could try instead
4. Be helpful and transparent about limitations

Provide a clear, honest answer based on what was successfully retrieved."""
        else:
            synthesis_context += "\nBased on the execution results above, provide a clear, natural language answer to the user's question. Be concise and accurate."
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful SEC financial data assistant. Synthesize the execution results into a clear, natural language answer. If there were failures, be transparent about what couldn't be retrieved."
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
