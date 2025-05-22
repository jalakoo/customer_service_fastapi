from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from simple_graph import query_graph

load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
RETRIEVER_MODEL = os.getenv("OPENAI_RETRIEVER_MODEL")
LLM_MODEL = os.getenv("OPENAI_MODEL")

app = FastAPI()

@app.post("/query_graph_only")
async def query_graph_only(query: str):
    try:
        response = query_graph(
            uri=URI, 
            username=AUTH[0], 
            password=AUTH[1], 
            retriever_model=RETRIEVER_MODEL, 
            llm_model=LLM_MODEL, 
            query=query
        )
        return {
            "answer": response.answer,
            "results": response.retriever_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
