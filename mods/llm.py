import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Literal
from mods.constants import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OPENAI_CHAT_MODEL
from pydantic.json_schema import JsonSchemaValue

from dotenv import load_dotenv
import ollama

load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """Abstract interface for LLM providers - allows swapping LLMs"""
        
    @abstractmethod
    def call(self, messages: List[Dict], tools: Optional[List[Dict]] = None, 
            tool_choice: str = "auto", *args, **kwargs) -> Dict[str, Any] | str:
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
    def __init__(self, model: str = DEFAULT_OPENAI_CHAT_MODEL, api_key: Optional[str] = None):
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
        print (f"Initialized OpenAI client with model: {self.model}")

    
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
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL):
        """
        Initialize Ollama client
        Args:
            model: Ollama model name
            api_url: Ollama API URL
        """
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.client = ollama.Client(host=self.base_url)
        print (f"Initialized Ollama client with model: {self.model} at {self.base_url}")


    def call(self,
             messages: List[Dict],
             tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto",
             stream: bool = False,
             response_format: JsonSchemaValue | Literal['', 'json'] | None = None
             ) -> str:
        """
        Calls the Ollama LLM with the provided messages and parameters.
        """
        try:
            if stream:
                full_response_content = []
                for chunk in self.client.chat(
                    model=self.model, messages=messages, stream=True, format=response_format
                ):
                    if chunk.get('done'):
                        break
    
                    content = chunk["message"]["content"]
                    print (content, end="", flush=True)
                    full_response_content.append(content)
                
                return "".join(full_response_content)
            else:
                response = self.client.chat(model=self.model, messages=messages, format=response_format)
                return response["message"]["content"]
        
        except ollama.ResponseError as e:
            print (f"Error interacting with Ollama: {e}")
            print (
                f"Ensure Ollama server is running and the model "
                f"'{self.model}' is dowloaded (`ollama pull {self.model}`)."
            )
            return f"Error: {e}"