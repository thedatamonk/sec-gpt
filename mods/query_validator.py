import json
from typing import Any, Dict, List, Optional

from mods.query_parser import SECQueryParser
from mods.utils import llm_call
from schemas.schema import CheckScopeSchema, FeasibilityCheckSchema
from templates.template import PromptTemplates


class QueryValidator:
    """
    QueryValidator is responsible for validating if the query is relevant to SEC filings,
    And if it's relevant it extract useful entities like company names, tickers, time period and financial metrics from the user query
    This info will be later used to formulate API calls to fetch data from SEC EDGAR database.
    """
    def __init__(self, stream: bool = False):
    
        self.stream = stream
        self.parser = SECQueryParser()

        
    def _check_scope(self, query: str) -> Dict[str, Any]:
        """
        This utility function checks if the query is within the financial domain using LLM and responds with a specific JSON format.
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
        """
        This utility function extracts and resolves entities like company names, tickers, time periods, and financial metrics from the user query.
        It uses regex-based parsing to do this.
        """

        try:
            parsed_data = self.parser.parse_query(query)
            return parsed_data.model_dump()

        except (json.JSONDecodeError, TypeError) as e:
            return {"Error": f"Could not parse any entities from the query: {e}"}

    def _check_feasibility(self, query: str, enriched_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        This utility function is used to check if the query is realistically answerable
        solely from information within public SEC filings (e.g., 10-K, 10-Q).
        Even though the query is in the financial scope, but not every query can be answered solely based on SEC filings.
        Hence this module is important to filter out such queries.

        NOTE: Currently, this function is not being used anywhere.
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
        The main query validation function that invokes other validation utility functions.
        """
        # STAGE 1: Check if the query is within the financial domain
        scope_status = self._check_scope(query)


        if not scope_status.get("is_related"):
             return {
                "status": "rejected",
                "message": "My purpose is to answer questions about public companies using SEC filings. Please ask a relevant question."
            }

        # STAGE 2: Extract and resolve entities from the query
        enriched_data = self._recognize_and_resolve_entities(query)
     
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


