from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.schema import get_schema
from validator import validate_answer_with_context
from variation import generate_query_variation
import os
import dotenv
dotenv.load_dotenv()

# Demo database credentials
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
RAG = None

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

    # Sample Schema
    # - **Staff**
    # - `id`: INTEGER Min: 1, Max: 6
    # - `name`: STRING Available options: ['Lena Patel', 'Marcus Lee', 'Anya Rojas', 'Tomoko Sato', 'Dennis Grant', 'Zhihao Chen']
    # - `bio`: STRING Available options: ['Former hardware engineer at a top semiconductor fi', 'Android developer with a background in mobile OS o', 'UI/UX designer specializing in mobile interfaces w', 'Electrical engineer with a PhD in signal processin', 'Product manager with 10 years of experience launch', 'Supply chain analyst with a background in electron']
    # - `hired_at`: DATE_TIME Min: 2024-11-30T12:15:00Z, Max: 2025-02-10T09:00:00Z
    # - **Doc**
    # - `id`: INTEGER Min: 1, Max: 6
    # - `title`: STRING Available options: ['Getting Started with Your Phone', 'Maximizing Battery Life', 'Customizing Your Device', 'Troubleshooting Common Issues', 'Using Advanced Camera Features', 'Securing Your Smartphone']
    # - `content`: STRING Available options: ['A quick-start guide covering unboxing', "Best practices for extending your smartphone's bat", 'Tips on personalizing settings', 'Step-by-step solutions for frequent problems like ', 'Guide to HDR', 'Instructions on enabling fingerprint']
    # - **Product**
    # - `id`: INTEGER Min: 1, Max: 3
    # - `name`: STRING Available options: ['Phoenix X1', 'Aurora Z5', 'Volt Mini']
    # - `description`: STRING Available options: ['Compact budget smartphone with dual cameras', 'Mid-range smartphone with AMOLED display and long ', 'Slim smartphone ideal for travel and backup use']
    # - `available_since`: DATE_TIME Min: 2025-02-28T09:30:00Z, Max: 2025-04-10T10:00:00Z
    # - `cost_usd`: INTEGER Min: 60, Max: 120
    # - `msrp_usd`: INTEGER Min: 130, Max: 300
    # - **Person**
    # - `id`: INTEGER Min: 1, Max: 10
    # - `name`: STRING Available options: ['Emily Carter', 'Jamal Nguyen', 'Sofia Bennett', 'Rajesh Verma', 'Mina Okafor', 'Bryce Lang', 'Linh Tran', 'Dario Costa', 'Keiko Yamamoto', 'Andres Silva']
    # - `email`: STRING Available options: ['emily.carter@technova.com', 'jamal.nguyen@innovamobile.com', 'sofia.bennett@nexondevices.com', 'rajesh.verma@globetel.com', 'mina.okafor@futurecom.com', 'bryce.lang@voltedge.com', 'linh.tran@omniwave.com', 'dario.costa@soniccell.com', 'keiko.yamamoto@skyreach.com', 'andres.silva@picochip.com']
    # - `customer_since`: DATE_TIME Min: 2024-06-22T09:00:00Z, Max: 2025-01-10T10:30:00Z
    # - **Company**
    # - `company_name`: STRING Available options: ['TechNova Inc', 'InnovaMobile', 'Nexon Devices', 'GlobeTel Solutions', 'FutureCom Ltd', 'VoltEdge Electronics', 'OmniWave Mobile', 'SonicCell Corp', 'SkyReach Tech', 'PicoChip Systems']
    # - **Order**
    # - `id`: INTEGER Min: 1, Max: 24
    # - `ordered_at`: DATE_TIME Min: 2025-05-10T03:55:09Z, Max: 2025-05-24T06:03:09Z
    # - `delivered_at`: DATE_TIME Min: 2025-05-14T05:03:09Z, Max: 2025-06-01T05:34:09Z
    # - **Vendor**
    # - `id`: INTEGER Min: 1, Max: 8
    # - `name`: STRING Available options: ['NanoTech Circuits', 'CrystalView Displays', 'PowerCore Batteries', 'HyperLink Antennas', 'TouchWave Sensors', 'VoltEdge Chargers', 'FlexiFrame Housings', 'OptiGlass Panels']
    # - **DELIVERY_SERVICE**
    # - `id`: INTEGER Min: 1, Max: 3
    # - `name`: STRING Available options: ['SwiftShip Logistics', 'AeroParcel Express', 'NeoTrack Couriers']
    # Relationship properties:
    # - **CONTAINS**
    # - `count`: INTEGER Min: 1, Max: 12
    # The relationships:
    # (:Staff)-[:MANAGES]->(:DELIVERY_SERVICE)
    # (:Staff)-[:OVERSEES]->(:Vendor)
    # (:Staff)-[:WORKS_FOR]->(:Staff)
    # (:Staff)-[:WROTE]->(:Doc)
    # (:Doc)-[:ABOUT]->(:Product)
    # (:Person)-[:WORKS_FOR]->(:Company)
    # (:Person)-[:MADE]->(:Order)
    # (:Order)-[:DELIVERED_BY]->(:DELIVERY_SERVICE)
    # (:Order)-[:CONTAINS]->(:Product)
    # (:Order)-[:ASSIGNED_TO]->(:Staff)
    # (:Vendor)-[:SUPPLIES]->(:Product)

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
            
            # Generate a valid aggregation query without using max(count())
            query = (
                f"MATCH (a:`{start_node}`)-[r:`{rel_type}`]->(b:`{end_node}`) "
                f"WITH a, count(r) AS rel_count "
                f"RETURN a, rel_count ORDER BY rel_count DESC LIMIT 5"
            )
            
            # Also add a simpler count query
            simple_count_query = (
                f"MATCH (a:`{start_node}`)-[r:`{rel_type}`]->(b:`{end_node}`) "
                f"RETURN a, count(r) AS rel_count ORDER BY rel_count DESC LIMIT 5"
            )
            
            # Add both variations
            queries = [query, simple_count_query]
        else:
            queries = [query]
            
        for q in queries:
            # Only add the query if it's valid
            if not validate_cypher_query(driver, q):
                continue
                
            examples.append(
                f"USER INPUT: 'Which {start_node} has the most {rel_type} relationships?' "
                f"QUERY: {q}"
            )
    
    # Add some additional safe aggregation examples
    safe_aggregation_queries = [
        "MATCH (n) RETURN labels(n) AS label, count(*) AS count ORDER BY count DESC LIMIT 10",
        "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY count DESC",
        "MATCH ()-[r]->() RETURN type(r) AS relationship_type, count(*) AS count ORDER BY count DESC",
    ]
    
    for q in safe_aggregation_queries:
        if validate_cypher_query(driver, q):
            examples.append(
                f"USER INPUT: 'Show me relationship type distribution' "
                f"QUERY: {q}"
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

    neo4j_schema = get_schema(driver, is_enhanced=True)
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
    query: str,
    retries: int = 0):
    
    # Connect to Neo4j database
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        rag = graphrag(
            driver=driver, 
            retriever_model=retriever_model,
            llm_model=llm_model)
        response = rag.search(
            query_text=query, 
            return_context=True)

        results = response.retriever_result.items
        if len(results) == 0:
            if retries < int(os.getenv("RETRY_COUNT")):
                print(f'No results found, retrying ({retries + 1}/{os.getenv("RETRY_COUNT")})')
                new_query = generate_query_variation(
                    query=query,
                    schema=get_schema(driver, is_enhanced=True),
                    prior_query=response.retriever_result.metadata["cypher"]
                )
                print(f'New query: {new_query}')
                return query_graph(
                    uri=uri, 
                    username=username, 
                    password=password,
                    retriever_model=retriever_model,
                    llm_model=llm_model,
                    query=new_query,
                    retries=retries + 1)
            else:
                return response
        
        # Check if answer matches with results
        is_valid = validate_answer_with_context(
            answer=response.answer,
            retrieved_results=response.retriever_result.items
        )
        print(f'Answer is valid check: {is_valid}')
        if not is_valid:
            print(f'Answer is not valid, retrying ({retries + 1}/{os.getenv("RETRY_COUNT")})')
            new_query = generate_query_variation(
                query=query,
                schema=get_schema(driver, is_enhanced=True),
                prior_query=response.retriever_result.metadata["cypher"]
            )
            return query_graph(
                uri=uri, 
                username=username, 
                password=password,
                retriever_model=retriever_model,
                llm_model=llm_model,
                query=new_query,
                retries=retries + 1)
        else:
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