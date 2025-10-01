import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from edgar import Company

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Standardized tool output"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseTool(ABC):
    """Base class for all tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
    
    def _handle_error(self, error: Exception, context: str = "") -> ToolResult:
        """Standard error handling"""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        logger.error(f"{self.name} error: {error_msg}")
        return ToolResult(success=False, error=error_msg)
    
class CompanyLookupTool(BaseTool):
    """Tool to find and validate company information"""
    
    @property
    def name(self) -> str:
        return "company_lookup"

    @property 
    def description(self) -> str:
        return "Find company information by ticker, or CIK"
    
    def execute(self, cik_or_ticker: str) -> ToolResult:
        """
        Look up company information
        Args:
            identifier: Company ticker, or CIK
        """
        try:
            cik_or_ticker = cik_or_ticker.strip().upper()
            
            # Try to get company info
            company = Company(cik_or_ticker)
            
            # Extract relevant info
            company_info = {
                "name": company.name,
                "cik": company.cik,
                "ticker": getattr(company, 'ticker', None),
                "sic": getattr(company, 'sic', None),
                "industry": getattr(company, 'industry', None),
            }
            
            return ToolResult(
                success=True, 
                data=company_info,
                metadata={"identifier_used": cik_or_ticker}
            )
        except Exception as e:
                if self._looks_like_ticker(cik_or_ticker):
                    return self._handle_error(e, f"Ticker '{cik_or_ticker}' not found")
                elif cik_or_ticker.isdigit():
                    return self._handle_error(e, f"CIK '{cik_or_ticker}' not found")
                else:
                    return self._handle_error(e, context="Invalid identifier format. Must be ticker or CIK")
    
    def _looks_like_ticker(self, text: str) -> bool:
        """Check if text looks like a stock ticker"""
        return len(text) <= 5 and text.isalpha()
    
class FinancialDataTool(BaseTool):
    """Tool to extract financial metrics from SEC filings"""
    
    @property
    def name(self) -> str:
        return "financial_data"
    
    @property
    def description(self) -> str:
        return "Get financial metrics (revenue, earnings, etc.) for a specific period"

    def execute(self, cik_or_ticker: str, year: int, quarter: Optional[int] = None, metric: str = "revenue") -> ToolResult:
        """
        Get financial data for a company
        Args:
            cik_or_ticker: Company ticker, or CIK
            year: Year (e.g., 2020)
            quarter: Quarter (1-4) or None for annual data
            metric: Financial metric to retrieve
        """
        try:
            # Get company
            company = Company(cik_or_ticker.strip().upper())

            # Determine filing type
            form_type = "10-Q" if quarter else "10-K"

            filings = company.get_filings(year=year, form=form_type)

            if not filings:
                return ToolResult(
                    success=False,
                    error=f"No {form_type} filings found for {company.name} in {year}"
                )
            
            # For quarterly data, find the right quarter
            target_filing = None
            if quarter:
                for filing in filings:
                    filing_quarter = self._extract_quarter_from_filing(filing)
                    if filing_quarter == quarter:
                        target_filing = filing
                        break
                        
                if not target_filing:
                    return ToolResult(
                        success=False,
                        error=f"No Q{quarter} {year} filing found for {company.name}"
                    )
            else:
                # For annual, get the most recent filing
                target_filing = filings[0]

            # Extract financial data
            financial_data = self._extract_financial_metrics(target_filing, metric)
        
            return ToolResult(
                success=True,
                data=financial_data,
                metadata={
                    "company": company.name,
                    "filing_date": target_filing.filing_date,
                    "period": f"Q{quarter} {year}" if quarter else str(year),
                    "form_type": form_type,
                    "accession_number": target_filing.accession_number
                }
            )
    
        except Exception as e:
            return self._handle_error(e, f"Failed to get financial data for {cik_or_ticker}")
        

    def _extract_quarter_from_filing(self, filing) -> Optional[int]:
        """Extract quarter number from filing"""
        try:
            # Get the filing period from the filing
            if hasattr(filing, 'period_of_report'):
                period = filing.period_of_report
                month = period.month
                return (month - 1) // 3 + 1
        except:
            pass
        return None
    
    def _extract_financial_metrics(self, filing, metric: str) -> Dict[str, Any]:
        """Extract specific financial metrics from filing"""
        try:
            # Get financials from the filing
            financials = filing.financials
            
            # Map common metric names
            metric_mapping = {
                "revenue": ["Revenues", "Revenue", "Total Revenue", "Net Sales"],
                "net_income": ["Net Income", "Net Earnings", "Net Income (Loss)"],
                "total_assets": ["Total Assets"],
                "total_equity": ["Total Equity", "Total Stockholders' Equity"]
            }
            
            metric_lower = metric.lower()
            possible_names = metric_mapping.get(metric_lower, [metric])
            
            result = {}
            
            # Try to find the metric in income statement first
            if hasattr(financials, 'income_statement'):
                income_stmt = financials.income_statement
                for name in possible_names:
                    if hasattr(income_stmt, name):
                        value = getattr(income_stmt, name)
                        result[metric] = value
                        break
            
            # If not found, try balance sheet
            if not result and hasattr(financials, 'balance_sheet'):
                balance_sheet = financials.balance_sheet
                for name in possible_names:
                    if hasattr(balance_sheet, name):
                        value = getattr(balance_sheet, name)
                        result[metric] = value
                        break
            
            # If still not found, return available metrics
            if not result:
                result = {
                    "requested_metric": metric,
                    "error": f"Metric '{metric}' not found",
                    "available_metrics": self._get_available_metrics(financials)
                }
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to extract metrics: {str(e)}"}


    def _get_available_metrics(self, financials) -> List[str]:
        """Get list of available financial metrics"""
        metrics = []
        try:
            if hasattr(financials, 'income_statement'):
                metrics.extend(dir(financials.income_statement))
            if hasattr(financials, 'balance_sheet'):
                metrics.extend(dir(financials.balance_sheet))
        except:
            pass
        return [m for m in metrics if not m.startswith('_')]