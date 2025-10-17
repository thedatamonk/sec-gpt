import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from edgar import get_filings
from .tool_schema import tool_schema

from ..core.client import EdgarClient
from ..core.models import FilingInfo
from ..utils.exceptions import FilingNotFoundError
from .types import ToolResponse


class FilingsTools:
    """Tools for filing-related operations."""

    def __init__(self):
        self.client = EdgarClient()

    @tool_schema(
        description="Get recent SEC filings for a specific company or across all companies, filtered by form type and time period",
        identifier_description="Company ticker (e.g., AAPL) or CIK number. Leave empty to get filings across all companies",
        form_type_description="Type of SEC filing to retrieve (e.g., '10-K', '10-Q', '8-K') or list of form types",
        days_description="Number of days to look back for recent filings",
        limit_description="Maximum number of filings to return"
    )
    def get_recent_filings(
        self,
        identifier: Optional[str] = None,
        form_type: Optional[Union[str, List[str]]] = None,
        days: int = 30,
        limit: int = 50,
    ) -> ToolResponse:
        """Get recent filings for a company or across all companies."""
        try:
            if identifier:
                # Company-specific filings
                company = self.client.get_company(identifier)
                filings = company.get_filings(form=form_type)
            else:
                # Global filings using edgar-tools get_filings()
                filings = get_filings(form=form_type, count=limit)

            # Limit results
            filings_list = []
            for i, filing in enumerate(filings):
                if i >= limit:
                    break

                # Convert date fields to datetime objects if they're strings
                filing_date = filing.filing_date
                if isinstance(filing_date, str):
                    filing_date = datetime.fromisoformat(filing_date.replace("Z", "+00:00"))

                acceptance_datetime = getattr(filing, "acceptance_datetime", None)
                if isinstance(acceptance_datetime, str):
                    acceptance_datetime = datetime.fromisoformat(acceptance_datetime.replace("Z", "+00:00"))

                period_of_report = getattr(filing, "period_of_report", None)
                if isinstance(period_of_report, str):
                    period_of_report = datetime.fromisoformat(period_of_report.replace("Z", "+00:00"))

                filing_info = FilingInfo(
                    accession_number=filing.accession_number,
                    filing_date=filing_date,
                    form_type=filing.form,
                    company_name=filing.company,
                    cik=filing.cik,
                    file_number=getattr(filing, "file_number", None),
                    acceptance_datetime=acceptance_datetime,
                    period_of_report=period_of_report,
                )
                filings_list.append(filing_info.to_dict())

            return {"success": True, "filings": filings_list, "count": len(filings_list)}
        except Exception as e:
            return {"success": False, "error": f"Failed to get recent filings: {str(e)}"}

    @tool_schema(
        description="Get the full content and metadata of a specific SEC filing by accession number",
        identifier_description="Company ticker (e.g., AAPL) or CIK number",
        accession_number_description="SEC filing accession number (e.g., '0000320193-23-000077')"
    )
    def get_filing_content(self, identifier: str, accession_number: str) -> ToolResponse:
        """Get the content of a specific filing."""
        try:
            company = self.client.get_company(identifier)

            # Find the specific filing
            filing = None
            # NOTE: get_filings() method is causing rate limiting issues
            for f in company.get_filings():
                if f.accession_number.replace("-", "") == accession_number.replace("-", ""):
                    filing = f
                    break

            if not filing:
                raise FilingNotFoundError(f"Filing {accession_number} not found")

            # Get filing content
            content = filing.text()

            # For structured filings, get the data object
            filing_data = {}
            try:
                obj = filing.obj()
                if obj:
                    # Extract key information based on filing type
                    if filing.form == "8-K" and hasattr(obj, "items"):
                        filing_data["items"] = obj.items
                        filing_data["has_press_release"] = getattr(obj, "has_press_release", False)
                    elif filing.form in ["10-K", "10-Q"]:
                        filing_data["has_financials"] = True
                    elif filing.form in ["3", "4", "5"]:
                        filing_data["is_ownership"] = True
            except Exception:
                pass

            return {
                "success": True,
                "accession_number": filing.accession_number,
                "form_type": filing.form,
                "filing_date": filing.filing_date.isoformat(),
                "content": content[:50000] if len(content) > 50000 else content,  # Limit size
                "content_truncated": len(content) > 50000,
                "filing_data": filing_data,
                "url": filing.url,
            }
        except FilingNotFoundError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Failed to get filing content: {str(e)}"}

    @tool_schema(
        description="Analyze an 8-K filing to identify specific material events, items reported, and press releases",
        identifier_description="Company ticker (e.g., AAPL) or CIK number",
        accession_number_description="SEC filing accession number of the 8-K filing"
    )
    def analyze_8k(self, identifier: str, accession_number: str) -> ToolResponse:
        """Analyze an 8-K filing for specific events."""
        try:
            company = self.client.get_company(identifier)

            # Find the specific filing
            filing = None
            for f in company.get_filings(form="8-K"):
                if f.accession_number.replace("-", "") == accession_number.replace("-", ""):
                    filing = f
                    break

            if not filing:
                raise FilingNotFoundError(f"8-K filing {accession_number} not found")

            # Get the 8-K object
            eightk = filing.obj()

            analysis: Dict[str, Any] = {
                "date_of_report": datetime.strptime(eightk.date_of_report, "%B %d, %Y").isoformat()
                if hasattr(eightk, "date_of_report")
                else None,
                "items": getattr(eightk, "items", []),
                "events": {},
            }

            # Check for common 8-K items
            item_descriptions = {
                "1.01": "Entry into Material Agreement",
                "1.02": "Termination of Material Agreement",
                "2.01": "Completion of Acquisition or Disposition",
                "2.02": "Results of Operations and Financial Condition",
                "2.03": "Creation of Direct Financial Obligation",
                "3.01": "Notice of Delisting",
                "4.01": "Changes in Accountant",
                "5.01": "Changes in Control",
                "5.02": "Departure/Election of Directors or Officers",
                "5.03": "Amendments to Articles/Bylaws",
                "7.01": "Regulation FD Disclosure",
                "8.01": "Other Events",
            }

            for item_code, description in item_descriptions.items():
                if hasattr(eightk, "has_item") and eightk.has_item(item_code):
                    analysis["events"][item_code] = {"present": True, "description": description}

            # Check for press releases
            if hasattr(eightk, "has_press_release"):
                analysis["has_press_release"] = eightk.has_press_release
                if eightk.has_press_release and hasattr(eightk, "press_releases"):
                    analysis["press_releases"] = [pr for pr in list(eightk.press_releases)[:3]]

            return {"success": True, "analysis": analysis}
        except Exception as e:
            return {"success": False, "error": f"Failed to analyze 8-K: {str(e)}"}

    @tool_schema(
        description="Extract specific sections from 10-K or 10-Q filings such as Business, Risk Factors, MD&A, and Financial Statements",
        identifier_description="Company ticker (e.g., AAPL) or CIK number",
        accession_number_description="SEC filing accession number",
        form_type_description="Type of SEC filing (10-K or 10-Q)",
        form_type_enum=["10-K", "10-Q"]
    )
    def get_filing_sections(self, identifier: str, accession_number: str, form_type: str) -> ToolResponse:
        """Get specific sections from a filing."""
        try:
            company = self.client.get_company(identifier)

            # Find the filing
            filing = None
            for f in company.get_filings(form=form_type):
                if f.accession_number.replace("-", "") == accession_number.replace("-", ""):
                    filing = f
                    break

            if not filing:
                raise FilingNotFoundError(f"Filing {accession_number} not found")

            # Get filing object
            filing_obj = filing.obj()

            sections = {}

            # Extract sections based on form type
            if form_type in ["10-K", "10-Q"]:
                # Business sections
                if hasattr(filing_obj, "business"):
                    sections["business"] = str(filing_obj.business)[:10000]

                # Risk factors
                if hasattr(filing_obj, "risk_factors"):
                    sections["risk_factors"] = str(filing_obj.risk_factors)[:10000]

                # MD&A
                if hasattr(filing_obj, "mda"):
                    sections["mda"] = str(filing_obj.mda)[:10000]

                # Financial statements
                if hasattr(filing_obj, "financials"):
                    sections["has_financials"] = True

            return {
                "success": True,
                "form_type": form_type,
                "sections": sections,
                "available_sections": list(sections.keys()),
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get filing sections: {str(e)}"}

    
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
    tools = FilingsTools()
    response = tools.get_recent_filings(identifier="AAPL", form_type=["10-K", "10-Q"], limit=5)
    print(response)