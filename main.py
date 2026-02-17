from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

class UserInput(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "HERMANN engine is running"}

@app.post("/chat")
def chat(input: UserInput):
    return {
        "reply": f"HERMANN received: {input.message}"
    }

