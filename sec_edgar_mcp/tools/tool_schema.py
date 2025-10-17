import inspect
from typing import Dict, List, Union, get_origin, get_type_hints

# Type mapping for JSON Schema
TYPE_MAPPING = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

def _get_json_type(python_type) -> str:
    """Convert Python type to JSON Schema type"""
    origin = get_origin(python_type)
    if origin is list or origin is List:
        return "array"
    elif origin is dict or origin is Dict:
        return "object"
    elif origin is Union:
        # For Optional types, get the non-None type
        args = [arg for arg in python_type.__args__ if arg is not type(None)]
        if args:
            return _get_json_type(args[0])
    
    return TYPE_MAPPING.get(python_type, "string")


def tool_schema(description: str, **extra_params):
    """
    Smart decorator that auto-infers parameter schema from type hints.
    
    Args:
        description: Human-readable description of what the tool does
        **extra_params: Optional overrides for parameter descriptions/constraints
            Format: {param_name}_description, {param_name}_enum, etc.
    
    Example:
        @tool_schema(
            description="Get company CIK by ticker",
            ticker_description="Stock ticker symbol (e.g., AAPL)"
        )
        def get_cik_by_ticker(self, ticker: str) -> ToolResponse:
            pass
    """
    def decorator(func):
        # Get function signature and type hints
        sig = inspect.signature(func)
        try:
            type_hints = get_type_hints(func)
        except Exception:
            # If type hints fail, use empty dict
            type_hints = {}
        
        # Build parameters schema
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter
            if param_name == 'self':
                continue
            
            # Get type from type hints (default to str if not found)
            param_type = type_hints.get(param_name, str)
            json_type = _get_json_type(param_type)
            
            # Check if required (no default value)
            is_required = param.default == inspect.Parameter.empty
            if is_required:
                required.append(param_name)
            
            # Build property schema
            param_schema = {
                "type": json_type
            }
            
            # Add description (custom or auto-generated)
            custom_desc_key = f"{param_name}_description"
            if custom_desc_key in extra_params:
                param_schema["description"] = extra_params[custom_desc_key]
            else:
                # Auto-generate basic description
                param_schema["description"] = f"{param_name.replace('_', ' ').title()}"
            
            # Add enum if specified
            enum_key = f"{param_name}_enum"
            if enum_key in extra_params:
                param_schema["enum"] = extra_params[enum_key]
            
            # Add default if present
            if param.default != inspect.Parameter.empty:
                param_schema["default"] = param.default
            
            properties[param_name] = param_schema
        
        # Store schema as function attribute
        func.__tool_schema__ = {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        
        return func
    
    return decorator