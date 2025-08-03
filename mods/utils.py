import openlit
import ollama
from typing import Dict, Any, Literal, List
from pydantic.json_schema import JsonSchemaValue
from schemas.schema import CheckScopeSchema



# Define constants
DEFAULT_OLLAMA_MODEL = "llama3:instruct"
# DEFAULT_OLLAMA_MODEL = "deepseek-r1:1.5b"
OTLP_ENDPOINT = "http://127.0.0.1:4318"

openlit.init(
    otlp_endpoint=OTLP_ENDPOINT
)

# Ollama LLM call
def llm_call(messages: List[Dict[str, Any]],
             model_name: str = DEFAULT_OLLAMA_MODEL,
             stream: bool = False,
             response_format: JsonSchemaValue | Literal['', 'json'] | None = None) -> str:
    

    try:
        if stream:
            full_response_content = []
            for chunk in ollama.chat(
                model=model_name, messages=messages, stream=True, format=response_format
            ):
                if chunk.get('done'):
                    break
  
                content = chunk["message"]["content"]
                print (content, end="", flush=True)
                full_response_content.append(content)
            
            return "".join(full_response_content)
        else:
            response = ollama.chat(model=model_name, messages=messages, format=response_format)
            return response["message"]["content"]
    
    except ollama.ResponseError as e:
        print (f"Error interacting with Ollama: {e}")
        print (
            f"Ensure Ollama server is running and the model "
            f"'{model_name}' is dowloaded (`ollama pull {model_name}`)."
        )
        return f"Error: {e}"


if __name__ == "__main__":

    query = "How many colors are there in a rainbow?"
    response = llm_call(messages = [{"role": "user", "content": query}], stream=True, response_format=CheckScopeSchema.model_json_schema())
    response_obj = CheckScopeSchema.model_validate_json(response).model_dump()
    # print (response_obj['is_related'], response_obj['reason'])


    prompt_template = \
f"""
Is the following user query related to public companies, finance, the stock market, or SEC filings?
Respond with a JSON object containing two keys: "is_related" (boolean) and "reason" (string).

User Query: "{query}"
"""
    response = llm_call(messages = [{"role": "user", "content": prompt_template}], stream=True, response_format=CheckScopeSchema.model_json_schema())
    response_obj = CheckScopeSchema.model_validate_json(response).model_dump()
    # print (response_obj['is_related'], response_obj['reason'])
    

