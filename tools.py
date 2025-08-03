from abc import ABC, abstractmethod
from typing import Any
import time
import re

# --- BASE TOOL DEFINITION ---

class BaseTool(ABC):
    """Abstract base class for all tools."""
    name: str = "base_tool"
    description: str = "This is a base tool."

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Executes the tool with the given parameters."""
        pass


# --- CONCRETE TOOL IMPLEMENTATIONS ---

class SearchFilingsTool(BaseTool):
    """Tool to search for SEC filings."""
    name = "search_filings"
    description = "Searches for a company's SEC filing for a specific year and form type. Parameters: company_ticker, form_type, year."

    def execute(self, **kwargs: Any) -> str:
        company_ticker = kwargs.get('company_ticker')
        form_type = kwargs.get('form_type')
        year = kwargs.get('year')

        # TODO: actual tool logic has to come here
        print(f"TOOL: Searching for {form_type} for {company_ticker} in {year}...")
        time.sleep(1)
        return f"Filing ID: {company_ticker}_{form_type}_{year}"

class GetFilingContentTool(BaseTool):
    """Tool to get the content of an SEC filing."""
    name = "get_filing_content"
    description = "Retrieves the text content of a specific filing using its ID. Parameters: filing_id."

    def execute(self, **kwargs: Any) -> str:
        filing_id = kwargs.get('filing_id')
        print(f"TOOL: Getting content for {filing_id}...")
        time.sleep(1)
        if filing_id == "AAPL_10K_2023": return "Content for filing AAPL_10K_2023. Total Revenue: $383B."
        if filing_id == "AAPL_10K_2022": return "Content for filing AAPL_10K_2022. Total Revenue: $394B."
        if filing_id == "MSFT_10K_2023": return "Content for filing MSFT_10K_2023. Total Revenue: $211B."
        if filing_id == "MSFT_10K_2022": return "Content for filing MSFT_10K_2022. Total Revenue: $198B."
        return "Content not found."
    
class CalculateTool(BaseTool):
    """Tool for performing calculations."""
    name = "calculate"
    description = "Evaluates a mathematical expression to perform calculations, like percentage change. Parameters: expression."

    def execute(self, **kwargs: Any) -> str:
        expression = kwargs.get('expression', "")
        print(f"TOOL: Calculating '{expression}'...")
        time.sleep(1)
        try:
            # A simple, safer way to evaluate math expressions
            result = eval(expression, {"__builtins__": None}, {})
            return f"Calculation Result: {result:.2f}%"
        except Exception as e:
            return f"Error calculating: {e}"


class ExtractDataTool(BaseTool):
    """Tool for extracting information from text."""
    name = "extract_data"
    description = "Extracts a single, specific data point (e.g., 'Total Revenue') from a body of text. Parameters: content, data_point."

    def execute(self, **kwargs: Any) -> str:
        content = kwargs.get('content', "")
        data_point = kwargs.get('data_point', "")
        print(f"TOOL: Extracting '{data_point}' from content...")
        time.sleep(1)
        
        # CORRECTED: Create a dynamic regex based on the data_point parameter.
        # This looks for the data point followed by a colon and captures the value.
        regex = re.escape(data_point) + r":\s*(.*)"
        match = re.search(regex, content, re.IGNORECASE)
        
        if match:
            # Return the data point and its found value, stripping any trailing periods.
            extracted_value = match.group(1).strip().strip('.')
            return f"{data_point}: {extracted_value}"
        
        return f"Data point '{data_point}' not found in content."

