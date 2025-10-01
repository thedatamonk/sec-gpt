from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CheckScopeSchema(BaseModel):
    is_related: bool
    reason: Optional[str] = None

    @model_validator(mode='after')
    def check_reason_if_not_related(self) -> 'CheckScopeSchema':
          """
          Validates that 'reason' is provided if 'is_related' is False.
          """
          if not self.is_related and not self.reason:
              raise ValueError("'reason' cannot be empty when 'is_related' is False.")
          
          return self

class Company(BaseModel):
    """A model for a single company with a name and/or a ticker."""
    type: Literal["name", "cik", "ticker"]
    value: str


class FinancialMetrics(str, Enum):
    REVENUE = 'revenue'
    NET_INCOME = 'net_income'
    CASH_FLOW_FROM_OPERATING_ACTIVITIES = 'cash_flow_from_operating_activities'
    OPERATING_EXPENSES = 'operating_expenses'
    GROSS_PROFIT = 'gross_profit'
    COST_OF_GOODS_SOLD = 'cost_of_goods_sold'
    TOTAL_ASSETS = 'total_assets'
    TOTAL_LIABILITIES = 'total_liabilities'
    TOTAL_EQUITY = 'total_equity'
    EARNINGS_PER_SHARE = 'earnings_per_share'


class FinancialEntitiesSchema(BaseModel):
    """The main model for extracting financial entities from a query."""
    companies: List[Company] = Field(
        default_factory=list,
        description="A list of companies mentioned in the query, identified by name, CIK, or ticker."
    )
    financial_metrics: List[FinancialMetrics] = Field(
        default_factory=list,
        description="A list of financial metrics requested in the query, e.g., 'revenue' or 'net_income'."
    )
    time_period: List[str] = Field(
        default_factory=list,
        description="A list of time periods mentioned in the query, e.g., 'Q3 2023' or '2024'."
    )


class FeasibilityCheckSchema(BaseModel):
    """
    A model to check if a query is feasible to answer using SEC filings.
    """
    is_feasible: bool
    reason: str


class Action(BaseModel):
    """Defines the structure for a tool-based action."""
    tool_name: str = Field(..., description="The name of the tool to be called.")
    parameters: Dict[str, Any] = Field(..., description="The parameters to pass to the tool, as a dictionary. For the 'Finish' tool, the key should be 'answer'.")

class ReACTResponseSchema(BaseModel):
    """The overall JSON structure the LLM must output."""
    thought: str = Field(..., description="Your reasoning, analysis of the current situation, and plan for the next step.")
    action: Action


if __name__ == "__main__":
     print (CheckScopeSchema.model_json_schema())
     print (FinancialEntitiesSchema.model_json_schema())
     print (FeasibilityCheckSchema.model_json_schema())
     print (FinancialEntitiesSchema.model_json_schema())
     print (FeasibilityCheckSchema.model_json_schema())
