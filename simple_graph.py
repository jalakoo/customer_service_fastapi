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
RAG = None

def get_node_properties(driver):
    """Get node properties from the database schema.
    
    Args:
        driver: Neo4j database driver
        
    Returns:
        dict: Dictionary mapping labels to their properties
    """
    query = """
    CALL db.schema.nodeTypeProperties()
    """
    result = driver.execute_query(query)
    
    return result

def get_relationship_properties (driver):
    query = "CALL db.schema.relTypeProperties()"
    result = driver.execute_query(query)

    return result

def get_relationships(driver):
    query = """
    CALL db.schema.visualization()
    YIELD relationships
    UNWIND relationships AS rel
    RETURN 
      rel.start.labels[0] AS from,
      rel.relationshipType AS type,
      rel.end.labels[0] AS to
    """
    result = driver.execute_query(query)
    
    return result

def getSchema(driver: GraphDatabase):
    node_props = get_node_properties(driver)
    rels = get_relationships(driver)
    rel_props = get_relationship_properties(driver)

    result = "Nodes:\n"
    result += f"{node_props}\n"
    result += "\n"
    result += "Relationships:\n"
    result += f"{rels}\n"
    result += "\n"
    result += "Relationship Properties:\n"
    result += f"{rel_props}\n"
    
    return result

def validate_cypher_query(driver: GraphDatabase, query: str) -> bool:
    """Validate if a Cypher query is valid by attempting to run it with LIMIT 1.
    
    Args:
        driver: Neo4j database driver
        query: Cypher query to validate
        
    Returns:
        bool: True if query executes successfully, False otherwise
    """
    # Add LIMIT 1 if not already present to minimize impact
    test_query = query
    if 'LIMIT' not in query.upper() and 'RETURN' in query.upper():
        test_query = f"{query.rstrip(';').strip()} LIMIT 1"
    
    try:
        driver.execute_read(test_query)
        return True
    except Exception as e:
        print(f"Query validation failed for '{test_query}': {str(e)}")
        return False


def getExampleQueries(driver: GraphDatabase, schema: str) -> list[str]:
    """Generate example queries based on the provided schema.
    
    Args:
        driver: Neo4j database driver
        schema: String containing the database schema in a specific format
        
    Returns:
        list[str]: List of example queries in the format "USER INPUT: 'query' QUERY: cypher_query"
    """
    examples = []
    
    # Parse node properties
    node_props = {}
    rels = []
    
    # Split schema into lines and process
    lines = [line.strip() for line in schema.split('\n') if line.strip()]
    
    # Process node properties
    i = 0
    while i < len(lines) and 'Node properties:' not in lines[i]:
        i += 1
    i += 1  # Skip the 'Node properties:' line
    
    while i < len(lines) and not lines[i].startswith('The relationships:'):
        if '{' in lines[i] and '}' in lines[i]:
            label = lines[i].split('{')[0].strip()
            props_str = lines[i].split('{')[1].split('}')[0].strip()
            props = [p.split(':')[0].strip() for p in props_str.split(',')]
            node_props[label] = props
        i += 1
    
    # Process relationships
    i += 1  # Skip the 'The relationships:' line
    while i < len(lines):
        line = lines[i].strip()
        if ':' in line and '[' in line and ']' in line:
            # Extract relationship pattern
            rel_pattern = line.split(':', 1)[1].strip()
            rels.append(rel_pattern)
        i += 1
    
    # Generate example queries
    
    # 1. Basic node queries
    for label, props in node_props.items():
        if props:
            # Simple match query
            props_str = ', '.join([f"n.{p}" for p in props[:3]])  # Use first 3 properties
            query = f"MATCH (n:`{label}`) RETURN {props_str} LIMIT 5"
            examples.append(
                f"USER INPUT: 'Show me some {label} records' "
                f"QUERY: {query}"
            )
            
            # Count query
            query = f"MATCH (n:`{label}`) RETURN count(n) AS count"
            examples.append(
                f"USER INPUT: 'How many {label} records are there?' "
                f"QUERY: {query}"
            )
    
    # 2. Relationship-based queries
    for rel in rels:
        # Extract node labels and relationship type
        parts = rel.split(':')
        if len(parts) < 2:
            continue
            
        # Simple relationship query
        query = f"MATCH {rel} RETURN * LIMIT 5"
        examples.append(
            f"USER INPUT: 'Show me records connected by {parts[1]}' "
            f"QUERY: {query}"
        )
        
        # Aggregation query for relationships
        if '>' in rel and '<' in rel:  # Bidirectional
            continue
            
        if '->' in rel:
            # Directional relationship - can do counts
            start_node = rel.split('(')[1].split(':')[1].strip('`')
            end_node = rel.split(')')[-2].split(':')[-1].strip('`')
            rel_type = parts[1].split(']')[0].strip('[]')
            
            query = (
                f"MATCH (a:`{start_node}`)-[r:`{rel_type}`]->(b:`{end_node}`) "
                f"WITH a, count(r) AS rel_count "
                f"RETURN a, rel_count ORDER BY rel_count DESC LIMIT 5"
            )
            
            # Only add the query if it's valid
            if not validate_cypher_query(driver, query):
                continue
            examples.append(
                f"USER INPUT: 'Which {start_node} has the most {rel_type} relationships?' "
                f"QUERY: {query}"
            )
    


    return examples


def graphrag(
    driver: GraphDatabase,
    retriever_model: str,
    llm_model: str,
    ) -> GraphRAG:


    global RAG
    if RAG is not None:
        return RAG

    neo4j_schema = getSchema(driver)
    print(f'Auto-generated Schema: {neo4j_schema}')
    
    examples = getExampleQueries(driver, neo4j_schema)
    print(f'Auto-generated Examples: {examples}')

    # (Optional) Manually Specify your own Neo4j schema
    # neo4j_schema = """
    # Node properties:
    # Person {name: STRING, born: INTEGER}
    # Movie {tagline: STRING, title: STRING, released: INTEGER}
    # Relationship properties:
    # ACTED_IN {roles: LIST}
    # REVIEWED {summary: STRING, rating: INTEGER}
    # The relationships:
    # (:Person)-[:ACTED_IN]->(:Movie)
    # (:Person)-[:DIRECTED]->(:Movie)
    # (:Person)-[:PRODUCED]->(:Movie)
    # (:Person)-[:WROTE]->(:Movie)
    # (:Person)-[:FOLLOWS]->(:Person)
    # (:Person)-[:REVIEWED]->(:Movie)
    # """

    # (Optional) Provide user input/query pairs for the LLM to use as examples
    # examples = [
    # "USER INPUT: 'Which Agent was assigned the most Chats?' QUERY: MATCH (p:Agent)-[:ASSIGNED_TO]-(m:Chat) WITH p.name AS name, COUNT(m) AS chat_count RETURN name, chat_count ORDER BY chat_count DESC LIMIT 1"
    # ]


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
    RAG = GraphRAG(retriever=retriever, llm=llm)
    return RAG

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