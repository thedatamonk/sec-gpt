import requests
import json
from typing import Dict, Any, List, Optional
from schemas.schema import CheckScopeSchema, FinancialEntitiesSchema, FeasibilityCheckSchema
from mods.utils import llm_call
from templates.template import PromptTemplates
from mods.query_parser import SECQueryParser

class QueryValidator:
    """
        An LLM-powered module to validate, sanitize, and enrich user queries about SEC filings.

        This class now uses a Large Language Model (LLM) for core NLP tasks:
        1.  Initialization: Fetches a list of all public companies from the SEC to
            ground the LLM and perform final entity resolution.
        2.  Scope Check (LLM-based): Asks an LLM to classify if the query is relevant.
        3.  Entity Recognition (LLM-based): Instructs an LLM to extract companies,
            metrics, and dates in a structured JSON format.
        4.  Feasibility Check (LLM-based): Asks an LLM to determine if the query is
            answerable solely from data in SEC filings.
    """

    def __init__(self, stream: bool = False):
    
        self.stream = stream
        self.parser = SECQueryParser()

        
    def _check_scope(self, query: str) -> Dict[str, Any]:
        """
        Stage 1 (LLM-based): Checks if the query is within the financial domain.
        """
        prompt_template = \
f"""
Is the following user query related to public companies, finance, the stock market, or SEC filings?
Respond with a JSON object containing two keys: "is_related" (boolean) and "reason" (string).

User Query: "{query}"
"""
        messages = [{
            "role": "user",
            "content": prompt_template
        }]
        
        try:
            response = llm_call(messages=messages, response_format=CheckScopeSchema.model_json_schema(), stream=self.stream)
            response_obj = CheckScopeSchema.model_validate_json(response).model_dump()
            return response_obj
        except (json.JSONDecodeError, TypeError) as e:
            return {"Error": e}

    def _recognize_and_resolve_entities(self, query: str) -> Dict[str, Any]:

        try:
            # response = llm_call(messages=messages, response_format=FinancialEntitiesSchema.model_json_schema(), stream=self.stream)
            # extracted_entities = FinancialEntitiesSchema.model_validate_json(response).model_dump()
            # company_infos = extracted_entities.get("companies", [])
            parsed_data = self.parser.parse_query(query)
            return parsed_data.model_dump()

        except (json.JSONDecodeError, TypeError) as e:
            return {"Error": f"Could not parse any entities from the query: {e}"}

    def _check_feasibility(self, query: str, enriched_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3 (LLM-based): Checks if the query is realistically answerable
        solely from information within public SEC filings (e.g., 10-K, 10-Q).
        NOTE: We might have to use a much more intelligent LLM because here
        The LLM has to think what the SEC API can be used for.
        """
        prompt_template = PromptTemplates.SEC_FEASIBILITY_CHECK.substitute(query=query, enriched_data=json.dumps(enriched_data, indent=2))
        
        messages = [{
            "role": "user",
            "content": prompt_template
        }]


        try:
            response = llm_call(messages=messages, response_format=FeasibilityCheckSchema.model_json_schema(), stream=self.stream)
            response_obj = FeasibilityCheckSchema.model_validate_json(response).model_dump()

            return response_obj
        
        except (json.JSONDecodeError, TypeError) as e:
            return {"Error": f"There was an issue analyzing the feasibility of your query. {e}"}
    
    def validate_and_enrich(self, query: str, history: Optional[List] = None) -> Dict[str, Any]:
        """
        The main validation pipeline that processes a user query.
        """
        scope_status = self._check_scope(query)

        if not scope_status.get("is_related"):
             return {
                "status": "rejected",
                "message": "My purpose is to answer questions about public companies using SEC filings. Please ask a relevant question."
            }
        
        enriched_data = self._recognize_and_resolve_entities(query)
        # if "Error" in enriched_data.keys():
        #     return {
        #         "status": "entity_detection_failed",
        #         "message": "I can't find any useful entities like the public company, ticker name etc in your query. Can you please paraphrase the question to include all these details?"
        #     }

        # # If we have metrics but no company, ask for clarification first.
        # if len(enriched_data.get("financial_metrics")) > 0 and len(enriched_data.get("companies")) == 0:
        #     return {
        #         "status": "clarification_needed",
        #         "message": "It looks like you're asking about a financial metric, but I don't know which company to analyze. Could you please specify a company or ticker?",
        #         "enriched_query": enriched_data
        #     }
        
        # rejection_message = self._check_feasibility(query, enriched_data)
        # if not rejection_message.get("is_feasible"):
        #     return {"status": "rejected", "message": rejection_message.get("reason"), "enriched_query": enriched_data}
        

        # if len(enriched_data.get("companies")) > 0 and len(enriched_data.get("financial_metrics")) == 0:
        #      return {
        #         "status": "clarification_needed",
        #         "message": f"Could you specify which financial metric you're interested in?",
        #         "enriched_query": enriched_data
        #     }

        return {"status": "valid", "enriched_query": enriched_data}

if __name__ == "__main__":
    validator = QueryValidator(stream=True)

    queries_to_test = [
        # "What was the revenue for Ford Motor Company last year?",
        # "What is the best recipe for pasta carbonara?",
        "What's AAPL's profit for last year?",  # for this query, it's unable to extract entities; we need to improve the entity detection and resolution logic
        # "What were the risk factors?",  # We need to handle this case, when the question is ambiguous. in this case, ideally it should have responded with insufficient context please provide more info.
    ]
    print("\n--- Running Query Validation Tests with LLM-Powered Functions ---")
    for i, q in enumerate(queries_to_test):
        print(f"\n>>>>>>>>>>>>> Test Case {i+1} <<<<<<<<<<<<<")
        print(f"User Query: '{q}'")
        result = validator.validate_and_enrich(q)
        print(f"\nFinal Result for Query {i+1}:")
        print(f"    Status: {result['status']}")
        if 'message' in result:
            print(f"    Message: {result['message']}")
        if 'enriched_query' in result:
            print(f"    Enriched Data: {result.get('enriched_query')}")
    print("\n--- End of Tests ---")


