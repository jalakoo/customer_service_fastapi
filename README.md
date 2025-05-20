# Simple Text2Cyper GraphRAG
Simple FastAPI server and standalone simple_graph.py file for querying a Neo4j graph database using the [neo4j-graphrag]() package

## Config
Copy the .env.sample file to .env and fill in the values for your Neo4j and OpenAI credentials

## Running
```
uv sync
uv run uvicorn app:app --reload
```
FastAPI interactive docs will then be available at http://localhost:8000/docs