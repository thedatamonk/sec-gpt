from string import Template

class PromptTemplates:
    """Centralized prompt template management"""
    
    # Text generation templates

    QUERY_VALIDATION = Template("""
You are an expert financial data analyst. Your task is to analyze user queries about public company filings and identify specific financial metrics. Your analysis must be structured and follow these rules:

1. Strictly evaluate the query. 
2. Extract companies/CIK (Central Index Key)/ticker names from the query
3. Extract the financial metrics from the query.
4. Extract the time period for which the user is requesting info for.

Look at the examples carefully, and respond with a JSON object of predefined schema with the following keys:
1. "companies": A list of objects, where each object has "name" and/or "ticker" and/or "cik".
2. "financial_metrics": A list of standardised financial metric strings.
3. "time_period": A string describing the time period. (e.g., "last year", "Q3 2023", "2025", "H1 2024").

Examples:

Query: What's APPL revenue for financial year 2024?
Output:
{{
"companies": [
    {{"type": "ticker", "value": "APPL"}}
],
"financial_metrics": ["Revenue"],
"time_period": ["Financial Year 2024"]
}}

Query: Compare Apple and Nvidia's quarterly revenue and PnL from year 2024 to 2025.
Output:
{{
"companies": [
    {{"type": "name", "value": "Apple"}},
    {{"type": "name", "value": "Nvidia"}}
],
"financial_metrics": ["Quarterly Revenue", "Quarterly PnL"],
"time_period": ["2024", "2025"]
}}

Query: What's the annual turnover for the company whose cik is 0001652044?
Output:
{{
"companies": [
    {{"type": "cik", "value": "0001652044"}},
],
"financial_metrics": ["Annual Turnover"],
"time_period": []
}}
<END OF EXAMPLES>

User Query: "${query}"
""")
    

    SEC_FEASIBILITY_CHECK = Template("""
As an expert on SEC filings, is the user query below feasible to answer using ONLY data from public documents like 10-K, 10-Q, and 8-K reports?

Consider the following:
- The query must not be subjective (asking for opinions, feelings, or morale).
- The query must not be predictive (asking about future events or stock prices).
- The query must refer to a specific public company.
- The information must be something a company is required to report to the SEC.

Respond with a JSON object with two keys: "is_feasible" (boolean) and "reason" (string).

User Query: "${query}"
Context (Companies/Tickers/Time_Period/Financial Metrics extracted from the user query):
${enriched_data}
""")
    
    @classmethod
    def get_template(cls, template_name: str) -> Template:
        """Get a specific template by name"""
        return getattr(cls, template_name.upper(), None)
    
    @classmethod
    def list_templates(cls) -> list:
        """List all available templates"""
        return [attr for attr in dir(cls) if not attr.startswith('_') and isinstance(getattr(cls, attr), Template)]
