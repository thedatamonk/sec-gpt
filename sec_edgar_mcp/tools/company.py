import inspect
from typing import Dict, List

from ..core.client import EdgarClient
from ..core.models import CompanyInfo
from ..utils.exceptions import CompanyNotFoundError
from .tool_schema import tool_schema
from .types import ToolResponse


class CompanyTools:
    """Tools for company-related operations."""

    def __init__(self):
        self.client = EdgarClient()

    @tool_schema(
            description="Get the CIK (Central Index Key) for a company based on its ticker symbol",
            ticker_description="Company ticker symbol (e.g., AAPL, TSLA, MSFT)"
    )
    def get_cik_by_ticker(self, ticker: str) -> ToolResponse:
        """Get the CIK for a company based on its ticker symbol."""
        try:
            cik = self.client.get_cik_by_ticker(ticker)
            if cik:
                return {
                    "success": True,
                    "cik": cik,
                    "ticker": ticker.upper(),
                    "suggestion": f"Use CIK '{cik}' instead of ticker '{ticker}' for more reliable and faster API calls",
                }
            else:
                return {"success": False, "error": f"CIK not found for ticker: {ticker}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @tool_schema(
        description="Get detailed company information including CIK, name, ticker, SIC, industry classification, and fiscal year end",
        identifier_description="Company ticker (e.g., AAPL) or CIK number"
    )
    def get_company_info(self, identifier: str) -> ToolResponse:
        """Get detailed company information."""
        try:
            company = self.client.get_company(identifier)

            info = CompanyInfo(
                cik=company.cik,
                name=company.name,
                ticker=getattr(company, "tickers", [None])[0] if hasattr(company, "tickers") else None,
                sic=getattr(company, "sic", None),
                sic_description=getattr(company, "sic_description", None),
                exchange=getattr(company, "exchange", None),
                state=getattr(company, "state", None),
                fiscal_year_end=getattr(company, "fiscal_year_end", None),
            )

            return {"success": True, "company": info.to_dict()}
        except CompanyNotFoundError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Failed to get company info: {str(e)}"}

    @tool_schema(
        description="Search for companies by name and get a list of matching companies with their CIKs and tickers",
        query_description="Company name or partial name to search for",
        limit_description="Maximum number of results to return"
    )
    def search_companies(self, query: str, limit: int = 10) -> ToolResponse:
        """Search for companies by name."""
        try:
            results = self.client.search_companies(query, limit)

            companies = []
            for result in results:
                companies.append({"cik": result.cik, "name": result.name, "tickers": getattr(result, "tickers", [])})

            return {"success": True, "companies": companies, "count": len(companies)}
        except Exception as e:
            return {"success": False, "error": f"Failed to search companies: {str(e)}"}

    @tool_schema(
        description="Get comprehensive company facts and financial data including assets, liabilities, revenues, net income, EPS, cash, and other key GAAP metrics",
        identifier_description="Company ticker (e.g., AAPL) or CIK number"
    )
    def get_company_facts(self, identifier: str) -> ToolResponse:
        """Get company facts and financial data."""
        try:
            company = self.client.get_company(identifier)

            # Get company facts using edgar-tools
            facts = company.get_facts()

            if not facts:
                return {"success": False, "error": "No facts available for this company"}

            # Extract key financial metrics
            metrics = {}

            # Try to access the raw facts data
            if hasattr(facts, "data"):
                facts_data = facts.data

                # Look for US-GAAP facts
                if "us-gaap" in facts_data:
                    gaap_facts = facts_data["us-gaap"]

                    # Common metrics to extract
                    metric_names = [
                        "Assets",
                        "Liabilities",
                        "StockholdersEquity",
                        "Revenues",
                        "NetIncomeLoss",
                        "EarningsPerShareBasic",
                        "CashAndCashEquivalents",
                        "CommonStockSharesOutstanding",
                    ]

                    for metric in metric_names:
                        if metric in gaap_facts:
                            metric_data = gaap_facts[metric]
                            if "units" in metric_data:
                                # Get the most recent value
                                for unit_type, unit_data in metric_data["units"].items():
                                    if unit_data:
                                        # Sort by end date and get the latest
                                        sorted_data = sorted(unit_data, key=lambda x: x.get("end", ""), reverse=True)
                                        if sorted_data:
                                            latest = sorted_data[0]
                                            metrics[metric] = {
                                                "value": float(latest.get("val", 0)),
                                                "unit": unit_type,
                                                "period": latest.get("end", ""),
                                                "form": latest.get("form", ""),
                                                "fiscal_year": latest.get("fy", ""),
                                                "fiscal_period": latest.get("fp", ""),
                                            }
                                            break

            return {
                "success": True,
                "cik": company.cik,
                "name": company.name,
                "metrics": metrics,
                "has_facts": bool(facts),
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get company facts: {str(e)}"}


    @classmethod
    def get_tool_definitions(cls) -> List[Dict]:
        """
        Extract all decorated methods and return their OpenAI function schemas.
        
        Returns:
            List of tool definitions in OpenAI function calling format
        """
        definitions = []
        
        # Iterate through class methods
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Check if method has our schema decorator
            if hasattr(method, '__tool_schema__'):
                schema = method.__tool_schema__
                definitions.append({
                    "type": "function",
                    "function": schema
                })
        
        return definitions
    
    @classmethod
    def get_method_names(cls) -> List[str]:
        """
        Get list of all decorated method names.
        
        Returns:
            List of method names that have tool_schema decorator
        """
        method_names = []
        
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if hasattr(method, '__tool_schema__'):
                method_names.append(name)
        
        return method_names
    

if __name__ == "__main__":
    tools = CompanyTools()
    print(tools.get_cik_by_ticker("AAPL"))
    print(tools.get_company_info("AAPL"))
    print(tools.search_companies("Apple", limit=5))
    print(tools.get_company_facts("AAPL"))
    print(tools.get_company_facts("AAPL"))