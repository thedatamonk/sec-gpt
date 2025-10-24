from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import SecAgent
import asyncio

app = FastAPI(title="SEC Agent")


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

agent = SecAgent()

@app.get("/")
def read_root():
    return {"message": "SEC Agent is running!", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # this endpoint will be invoked when a user is chatting with SEC Agent

        response = await asyncio.to_thread(agent.run, request.message)

        return ChatResponse(response=response["content"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}