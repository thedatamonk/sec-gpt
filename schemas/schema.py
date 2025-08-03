from pydantic import BaseModel, model_validator, Field
from typing import Optional, Dict, Any



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
    name: Optional[str] = None
    ticker: Optional[str] = None

    @model_validator(mode='after')
    def check_at_least_one_field_provided(self) -> 'Company':
        """Ensures that at least a name or a ticker is given."""
        if not self.name and not self.ticker:
            raise ValueError("A Company object must have at least a 'name' or a 'ticker'.")
        return self

class FinancialEntitiesSchema(BaseModel):
    """The main model for extracting financial entities from a query."""
    companies: list[Company]
    metrics: list[str]
    period: str



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
