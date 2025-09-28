from pydantic import BaseModel
from typing import List
from schemas.schema import FinancialMetrics, Company
import re
import requests
from constants import SEC_COMPANY_TICKERS_URL, SEC_API_USER_AGENT
from pathlib import Path
from datetime import datetime, timedelta
import json


class ParsedQuery(BaseModel):
    companies: List[Company]
    financial_metrics: List[FinancialMetrics]
    time_period: List[str]


class SECQueryParser:
    def __init__(self, force_refresh: bool = False):
        # Live SEC company data structures
        self.ticker_to_cik = {}
        self.name_to_cik = {}
        self.cik_to_metadata = {}


        # Cache configuration
        self.cache_dir = Path(__file__).parent.parent / "cache"
        self.cache_file = self.cache_dir / "sec_company_data.json"
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_expiry_days = 7  # Cache expires after 7 days

        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load live SEC data instead of hardcoded mapping
        self._load_sec_company_data(force_refresh=force_refresh)

        # Financial metrics patterns
        self.metric_patterns = {
            FinancialMetrics.REVENUE: [
                r'\brevenue\b', r'\bsales\b', r'\btop.?line\b', r'\bgross.?sales\b'
            ],
            FinancialMetrics.NET_INCOME: [
                r'\bnet.?income\b', r'\bprofit\b', r'\bearnings\b', r'\bbottom.?line\b'
            ],
            FinancialMetrics.CASH_FLOW_FROM_OPERATING_ACTIVITIES: [
                r'\bcash.?flow\b', r'\boperating.?cash\b', r'\bfree.?cash.?flow\b'
            ],
            FinancialMetrics.TOTAL_ASSETS: [
                r'\btotal.?assets\b', r'\bassets\b'
            ],
            FinancialMetrics.OPERATING_EXPENSES: [
                r'\boperating.?expenses\b', r'\boperating.?costs\b', r'\bOPEX\b'
            ]
        }

        # Time period patterns
        self.time_patterns = {
            r'\bQ[1-4]\s+\d{4}\b': 'quarter',
            r'\b\d{4}\b': 'year',
            r'\blast\s+year\b': 'last_year',
            r'\blast\s+quarter\b': 'last_quarter',
            r'\bFY\s*\d{4}\b': 'fiscal_year',
            r'\byear\s+ending\b': 'year_ending'
        }

        # Patterns for different identifier types
        self.ticker_pattern = r'\b[A-Z]{1,5}\b'
        self.cik_pattern = r'\b(?:CIK[:\s]*)?(\d{1,10})\b'

    def _is_cache_valid(self) -> bool:
        """Check if the cache exists and is still valid (not expired)"""
        if not self.cache_file.exists() or not self.cache_metadata_file.exists():
            return False
        
        try:
            with open(self.cache_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cache_time = datetime.fromisoformat(metadata['last_updated'])
            expiry_time = cache_time + timedelta(days=self.cache_expiry_days)
            
            return datetime.now() < expiry_time
        except (json.JSONDecodeError, KeyError, ValueError):
            return False
        
    def _load_from_cache(self) -> dict:
        """Load company data from local cache"""
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            with open(self.cache_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"Loaded cached data for {metadata['company_count']} companies "
                  f"(cached on {metadata['last_updated'][:10]})")
            
            return data
        except Exception as e:
            print(f"Error loading from cache: {e}")
            return {}
    
    def _save_to_cache(self, data: dict):
        """Save company data to local cache"""
        try:
            # Save the actual data
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save metadata about the cache
            metadata = {
                'last_updated': datetime.now().isoformat(),
                'company_count': len(data.get('cik_to_metadata', {})),
                'cache_version': '1.0'
            }
            
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Successfully cached data for {metadata['company_count']} companies.")
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def _fetch_from_api(self) -> dict:
        """Fetch fresh data from SEC API"""
        print("Fetching fresh data from SEC API...")
        try:
            response = requests.get(SEC_COMPANY_TICKERS_URL, 
                                  headers={"User-Agent": SEC_API_USER_AGENT})
            response.raise_for_status()
            all_companies = response.json()
            
            # Process the data into our format
            ticker_to_cik = {}
            name_to_cik = {}
            cik_to_metadata = {}
            
            for company_data in all_companies.values():
                cik_str = str(company_data['cik_str']).zfill(10)
                ticker = company_data['ticker'].lower()
                name = company_data['title'].lower()
                
                if cik_str in cik_to_metadata:
                    cik_to_metadata[cik_str]["ticker"].append(ticker)
                else:
                    cik_to_metadata[cik_str] = {
                        "name": name,
                        "cik": cik_str,
                        "ticker": [ticker]
                    }
                
                ticker_to_cik[ticker] = cik_str
                name_to_cik[name] = cik_str
            
            # Structure data for caching
            processed_data = {
                'ticker_to_cik': ticker_to_cik,
                'name_to_cik': name_to_cik,
                'cik_to_metadata': cik_to_metadata
            }
            
            print(f"Successfully fetched data for {len(cik_to_metadata)} companies from API.")
            return processed_data
            
        except Exception as e:
            print(f"Error fetching from SEC API: {e}")
            return {}

    def _load_sec_company_data(self, force_refresh: bool = False):
        """
        Load SEC company data from cache or API.
        
        Args:
            force_refresh (bool): If True, bypass cache and fetch from API
        """
        data = {}
        if not force_refresh and self._is_cache_valid():
            # Load from cache
            data = self._load_from_cache()
        
        if not data:  # Cache invalid, doesn't exist, or force refresh
            # fetch from SEC API
            data = self._fetch_from_api()

            # Save to cache if we got data
            if data:
                self._save_to_cache(data)

        # Update instance variables
        if data:
            self.ticker_to_cik = data.get('ticker_to_cik', {})
            self.name_to_cik = data.get('name_to_cik', {})
            self.cik_to_metadata = data.get('cik_to_metadata', {})
        else:
            # Fallback to empty dictionaries if everything fails
            print("Warning: Could not load company data from cache or API")
            self.ticker_to_cik = {}
            self.name_to_cik = {}
            self.cik_to_metadata = {}

    def refresh_cache(self):
        """Public method to force refresh the cache"""
        print("Force refreshing SEC company data...")
        self._load_sec_company_data(force_refresh=True)
    
    def get_cache_info(self) -> dict:
        """Get information about the current cache"""
        if not self.cache_metadata_file.exists():
            return {"status": "No cache found"}
        
        try:
            with open(self.cache_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cache_time = datetime.fromisoformat(metadata['last_updated'])
            expiry_time = cache_time + timedelta(days=self.cache_expiry_days)
            is_valid = datetime.now() < expiry_time
            
            return {
                "status": "valid" if is_valid else "expired",
                "last_updated": metadata['last_updated'],
                "company_count": metadata['company_count'],
                "expires_at": expiry_time.isoformat(),
                "cache_file_size": f"{os.path.getsize(self.cache_file) / 1024:.1f} KB"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def extract_companies(self, query: str) -> List[Company]:
        """Extract and normalize company identifiers from query"""
        companies = []
        query_lower = query.lower()
        
        # 1. First check for CIK numbers (highest priority for SEC)
        cik_matches = re.findall(self.cik_pattern, query, re.IGNORECASE)
        for cik in cik_matches:
            # Try both padded and unpadded versions
            padded_cik = cik.zfill(10)  # Pad to 10 digits
            unpadded_cik = cik.lstrip('0')  # Remove leading zeros
            
            if padded_cik in self.cik_to_metadata:
                companies.append(Company(
                    type='cik',
                    value=self.cik_to_metadata[padded_cik]['name'].title()
                ))
            elif any(unpadded_cik == cik_key.lstrip('0') for cik_key in self.cik_to_metadata.keys()):
                # Find the matching padded CIK
                matching_cik = next(cik_key for cik_key in self.cik_to_metadata.keys() 
                                  if unpadded_cik == cik_key.lstrip('0'))
                companies.append(Company(
                    type='cik',
                    value=self.cik_to_metadata[matching_cik]['name'].title()
                ))
        
        # 2. Check for ticker symbols if no CIK found
        if not companies:
            tickers = re.findall(self.ticker_pattern, query)
            for ticker in tickers:
                ticker_lower = ticker.lower()
                if ticker_lower in self.ticker_to_cik:
                    cik = self.ticker_to_cik[ticker_lower]
                    companies.append(Company(
                        type='ticker',
                        value=self.cik_to_metadata[cik]['name'].title()
                    ))
        
        # 3. Check for company names if no tickers/CIKs found
        if not companies:
            # Check for exact or partial name matches
            for company_name, cik in self.name_to_cik.items():
                if company_name in query_lower:
                    companies.append(Company(
                        type='name',
                        value=self.cik_to_metadata[cik]['name'].title()
                    ))
                    break  # Take first match to avoid duplicates
        
        return companies

    def extract_financial_metrics(self, query: str) -> List[FinancialMetrics]:
        """Extract financial metrics from query"""
        metrics = []
        query_lower = query.lower()
        
        for metric, patterns in self.metric_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    metrics.append(metric)
                    break
        
        return metrics

    def get_company_info(self, company: Company):
        raise NotImplementedError("This method is not implemented yet.")

    def extract_time_periods(self, query: str) -> List[str]:
        """Extract time period information from query"""
        periods = []
        
        for pattern, period_type in self.time_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                periods.append(match.strip())
        
        return periods if periods else ['current']

    def parse_query(self, query: str) -> ParsedQuery:
        """Main parsing function"""
        companies = self.extract_companies(query)
        metrics = self.extract_financial_metrics(query)
        time_periods = self.extract_time_periods(query)

        return ParsedQuery(
            companies=companies,
            financial_metrics=metrics,
            time_period=time_periods
        )

if __name__ == "__main__":
    parser = SECQueryParser()

    # Test queries
    test_queries = [
        "Show me the revenue of Apple in Q2 2025",
        "What's AAPL's profit for last year?",
        "Get Microsoft's cash flow for FY2024",
        "TSLA operating income 2025",
        "Revenue for CIK 0000320193 in 2024",  # Apple's CIK
        "What's the net income of 1652044?",    # Alphabet's CIK (unpadded)
        "Show NVDA earnings for Q4 2024",
        "Get 320193 financial data",            # Apple's CIK (no padding)
        "Facebook revenue last quarter",
        "Show me CIK: 1318605 cash flow"       # Tesla's CIK
    ]
    
    for query in test_queries:
        result = parser.parse_query(query)
        print(f"Query: {query}")
        print(f"Result: {result.model_dump()}")
        print("-" * 50)