import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """Abstract interface for LLM providers - allows swapping LLMs"""
        
    @abstractmethod
    def call(self, messages: List[Dict], tools: Optional[List[Dict]] = None, 
            tool_choice: str = "auto") -> Dict[str, Any]:
        """
        Make LLM call with optional tool usage
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of tool definitions in OpenAI format
            tool_choice: 'auto', 'none', or specific tool name
        Returns:
            Dict with 'content' and optional 'tool_calls'
        """
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM"""
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize OpenAI client
        Args:
            model: Model name (gpt-3.5-turbo, gpt-4, etc.)
            api_key: OpenAI API key (if None, uses environment variable)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        
        self.model = model
        self.client = OpenAI(api_key=api_key or str(os.getenv("OPENAI_API_KEY")))

    
    def call(self, messages: List[Dict], tools: Optional[List[Dict]] = None, tool_choice: str = "auto") -> Dict[str, Any]:
        """Call OpenAI API with function calling support"""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 1.0
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            
            result = {
                "content": message.content,
                "tool_calls": [],
                "raw_tool_calls": []
            }

            if hasattr(message, 'tool_calls') and message.tool_calls:
                import json
                for tool_call in message.tool_calls:
                    # Parsed format for execution
                    result["tool_calls"].append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    })

                    # Original format for OpenAI messages
                    result["raw_tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                    )
            
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

class OllamaLLM(BaseLLM):
    """Ollama LLM"""
    def __init__(self):
        """
        Initialize Ollama client
        Args:
            model: Ollama model name
            api_url: Ollama API URL
        """
        raise NotImplementedError("Ollama LLM integration is not implemented yet.")

    
    def call(self, messages: List[Dict], tools: Optional[List[Dict]] = None, tool_choice: str = "auto") -> Dict[str, Any]:
        """Call Ollama API (no function calling support)"""
        raise NotImplementedError("Ollama LLM integration is not implemented yet.")