from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
import os
import dotenv
dotenv.load_dotenv()

# Demo database credentials
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))


def graphrag(
    driver: GraphDatabase,
    retriever_model: str,
    llm_model: str,
    ) -> GraphRAG:

    # (Optional) Specify your own Neo4j schema
    neo4j_schema = """
    Node properties:
    Agent {id: INTEGER, name: STRING, bio: STRING, hired_at: DATETIME}
    Customer {id: INTEGER, name: STRING, email: STRING, customer_since: DATETIME}
    Chat {id: INTEGER, title: STRING, content: STRING, opened_at: DATETIME, closed_at: DATETIME}
    The relationships:
    (:Agent)<-[:ASSIGNED_TO]-(:Chat)
    (:Customer)-[:STARTED]->(:Chat)
    """
    # (Optional) Provide user input/query pairs for the LLM to use as examples
    examples = [
    "USER INPUT: 'Which Agent was assigned the most Chats?' QUERY: MATCH (p:Agent)-[:ASSIGNED_TO]-(m:Chat) WITH p.name AS name, COUNT(m) AS chat_count RETURN name, chat_count ORDER BY chat_count DESC LIMIT 1"
    ]


    # Create LLM object
    t2c_llm = OpenAILLM(model_name=retriever_model)

    # Initialize the retriever
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=t2c_llm,
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    # Create LLM object
    llm = OpenAILLM(
        model_name=llm_model, 
        model_params={"temperature": 0})

    # Initialize the RAG pipeline
    return GraphRAG(retriever=retriever, llm=llm)

def query_graph(
    uri: str, 
    username: str, 
    password: str,
    retriever_model: str,
    llm_model: str,
    query: str):
    
    # Connect to Neo4j database
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        rag = graphrag(
            driver=driver, 
            retriever_model=retriever_model,
            llm_model=llm_model)
        response = rag.search(
            query_text=query, 
            return_context=True)
        return response

if __name__ == "__main__":
    response = query_graph(
        uri=URI, 
        username=AUTH[0], 
        password=AUTH[1], 
        retriever_model=os.getenv("OPENAI_RETRIEVER_MODEL"), 
        llm_model=os.getenv("OPENAI_MODEL"), 
        query="Which Agent was assigned the most Chats?"
    )
    print(response)