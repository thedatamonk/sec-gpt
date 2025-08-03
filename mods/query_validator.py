import requests
import json
from typing import Dict, Any, List, Optional
from schemas.schema import CheckScopeSchema, FinancialEntitiesSchema, FeasibilityCheckSchema
from mods.utils import DEFAULT_OLLAMA_MODEL, llm_call

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

    SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    def __init__(self, stream: bool = False):
        """
        Initializes the validator, loading the spaCy NLP model and fetching
        live company data from the SEC.
        """
        # spaCy can be kept as a potential fallback or for other NLP tasks.
        # try:
        #     self.nlp = spacy.load("en_core_web_sm")
        # except OSError:
        #     print("spaCy model 'en_core_web_sm' not found.")
        #     self.nlp = None

        # --- Live Entity Resolution Data ---
        # This data is still crucial to ground the LLM and resolve names to CIKs.
        self.ticker_to_cik = {}
        self.name_to_cik = {}
        self.cik_to_metadata = {}
        self.stream = stream
        self._load_sec_company_data()

    def _load_sec_company_data(self):
        """
        Fetches and processes the company ticker and CIK data from the SEC.
        This data will be used to verify entities identified by the LLM.
        """
        print("Loading company data from SEC...")
        try:
            response = requests.get(self.SEC_COMPANY_TICKERS_URL, headers={"User-Agent": "rohilpal9763@gmail.com"})
            response.raise_for_status()
            all_companies = response.json()
            for company_data in all_companies.values():
                cik_str = str(company_data['cik_str']).zfill(10)
                ticker = company_data['ticker']
                name = company_data['title']
                self.cik_to_metadata[cik_str] = {"name": name, "cik": cik_str, "ticker": ticker}
                self.ticker_to_cik[ticker.lower()] = cik_str
                self.name_to_cik[name.lower()] = cik_str
            print(f"Successfully loaded data for {len(self.cik_to_metadata)} companies.")
        except Exception as e:
            print(f"Error fetching/processing SEC data: {e}.")

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
        """
        Stage 2 (LLM-based): Uses an LLM to extract entities and then resolves
        them using the loaded SEC data.
        """
        prompt_template = \
f"""
Extract the entities from the user query below.
Respond with a single, minified JSON object with three keys:
1. "companies": A list of objects, where each object has "name" and/or "ticker".
2. "metrics": A list of financial metric strings (e.g., "revenue", "debt", "10-K risk factors").
3. "period": A string describing the time period (e.g., "last year", "Q3 2023").

User Query: "{query}"
"""

        messages = [{
            "role": "user",
            "content": prompt_template
        }]
        enriched_data = {"entities": [], "metrics": [], "period": None}
        resolved_ciks = set()

        try:
            response = llm_call(messages=messages, response_format=FinancialEntitiesSchema.model_json_schema(), stream=self.stream)
            extracted_entities = FinancialEntitiesSchema.model_validate_json(response).model_dump()
            llm_companies = extracted_entities.get("companies", [])
            
            for company in llm_companies:
                ticker = company.get("ticker", "").lower() if company.get("ticker") else ""
                name = company.get("name", "").lower() if company.get("name") else ""
                
                found_cik = None

                # Priority 1: Ticker match (most reliable)
                if ticker and ticker in self.ticker_to_cik:
                    found_cik = self.ticker_to_cik[ticker]
                
                # Priority 2: Exact name match
                elif name and name in self.name_to_cik:
                    found_cik = self.name_to_cik[name]
                
                # Priority 3: Partial name match (fallback for cases like "Apple" vs "Apple Inc.")
                elif name:
                    for official_name, cik_val in self.name_to_cik.items():
                        if name.lower() in official_name.lower():
                            found_cik = cik_val
                            break # Take the first partial match
                
                if found_cik and found_cik not in resolved_ciks:
                    enriched_data["entities"].append(self.cik_to_metadata[found_cik])
                    resolved_ciks.add(found_cik)


            enriched_data["metrics"] = extracted_entities.get("metrics", [])
            enriched_data["period"] = extracted_entities.get("period")
            return enriched_data
        
        except (json.JSONDecodeError, TypeError) as e:
            return {"Error": f"Could not parse LLM response for entity extraction: {e}"}


    def _check_feasibility(self, query: str, enriched_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3 (LLM-based): Checks if the query is realistically answerable
        solely from information within public SEC filings (e.g., 10-K, 10-Q).
        NOTE: We might have to use a much more intelligent LLM because here
        The LLM has to think what the SEC API can be used for.
        """
        prompt_template = \
f"""
As an expert on SEC filings, is the user query below feasible to answer using ONLY data from public documents like 10-K, 10-Q, and 8-K reports?

Consider the following:
- The query must not be subjective (asking for opinions, feelings, or morale).
- The query must not be predictive (asking about future events or stock prices).
- The query must refer to a specific public company.
- The information must be something a company is required to report to the SEC.

Respond with a JSON object with two keys: "is_feasible" (boolean) and "reason" (string).

User Query: "{query}"
Context (enriched entities found): {json.dumps(enriched_data)}
"""
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
        if "Error" in enriched_data.keys():
            return {
                "status": "entity_detection_failed",
                "message": "I can't find any useful entities like the public company, ticker name etc in your query. Can you please paraphrase the question to include all these details?"
            }

        # If we have metrics but no company, ask for clarification first.
        if enriched_data.get("metrics") and not enriched_data.get("entities"):
            return {
                "status": "clarification_needed",
                "message": "It looks like you're asking about a financial metric, but I don't know which company to analyze. Could you please specify a company or ticker?",
                "enriched_query": enriched_data
            }
        
        rejection_message = self._check_feasibility(query, enriched_data)
        if not rejection_message.get("is_feasible"):
            return {"status": "rejected", "message": rejection_message.get("reason"), "enriched_query": enriched_data}
        

        if enriched_data["entities"] and not enriched_data["metrics"]:
             return {
                "status": "clarification_needed",
                "message": f"I see you're asking about {enriched_data['entities'][0]['name']}. Could you specify which financial metric you're interested in?",
                "enriched_query": enriched_data
            }

        return {"status": "valid", "enriched_query": enriched_data}


if __name__ == "__main__":
    validator = QueryValidator(stream=True)

    if validator.cik_to_metadata:
        queries_to_test = [
            "What was the revenue for Ford Motor Company (F) last year?",
            "What is the best recipe for pasta carbonara?",
            "Will Apple's stock price double next year?",  # for this query, it's unable to extract entities; we need to improve the entity detection and resolution logic
            "What were the risk factors?",  # We need to handle this case, when the question is ambiguous. in this case, ideally it should have responded with insufficient context please provide more info.
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

