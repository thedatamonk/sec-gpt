from string import Template
from typing import Dict, Any

def render_prompt(template: Template, **kwargs) -> str:
    """Render a template with the provided variables"""
    try:
        return template.substitute(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required template variable: {e}")

def validate_template_vars(template: Template, provided_vars: Dict[str, Any]) -> bool:
    """Validate that all required template variables are provided"""
    import re
    required_vars = set(re.findall(r'\$\{(\w+)\}', template.template))
    provided_vars_set = set(provided_vars.keys())
    missing_vars = required_vars - provided_vars_set
    
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    return True