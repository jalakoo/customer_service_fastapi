import os
import json
from openai import OpenAI
from typing import List, Dict, Any


def generate_answer_variation(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    model: str = "gpt-4"
) -> str:
    """
    Generates an answer to the query based on the retrieved results using OpenAI.
    
    Args:
        query (str): The original query to answer
        retrieved_results (List[Dict]): List of retrieved context items
        model (str): The OpenAI model to use for generation
        
    Returns:
        str: A generated answer based on the retrieved results
    """
    if not retrieved_results:
        return "I couldn't find enough information to answer that question."
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare the context from retrieved results
    def format_result(result: Dict[str, Any]) -> str:
        """Format a single result item for the prompt."""
        try:
            # Try to extract relevant fields
            content = result.get('content') or result.get('text') or str(result)
            # If it's a string representation of a dict, try to parse it
            if isinstance(content, str) and content.startswith('{') and content.endswith('}'):
                try:
                    content = json.dumps(json.loads(content), indent=2)
                except:
                    pass
            return content
        except Exception as e:
            print(f"Error formatting result: {e}")
            return str(result)
    
    # Create a clean context string
    context = "\n\n".join([
        f"--- Source {i+1} ---\n{format_result(result)}"
        for i, result in enumerate(retrieved_results)
    ])
    
    # Create the prompt
    prompt = f"""
    You are a helpful assistant that answers questions based on the provided context.
    
    QUESTION:
    {query}
    
    CONTEXT:
    {context}
    
    INSTRUCTIONS:
    1. Answer the question using ONLY the information from the provided context.
    2. If the context doesn't contain enough information to answer the question, 
       clearly state that you don't have enough information.
    3. Be concise and to the point.
    4. Do not make up any information that's not in the context.
    5. If the context contains multiple sources, synthesize the information 
       to provide a comprehensive answer.
    
    YOUR ANSWER:
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more focused answers
            max_tokens=500
        )
        
        # Extract and clean the response
        answer = response.choices[0].message.content.strip()
        
        # Ensure the answer is not empty
        if not answer:
            return "I couldn't generate an answer based on the available information."
            
        return answer
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return "I encountered an error while generating an answer."



def generate_query_variation(
    query: str,
    schema: str,
    prior_query: str,
    model: str = "gpt-4"
) -> str:
    """
    Generates a natural language variation of the original query while maintaining the same intent.
    
    Args:
        query (str): The original query to create a variation of
        schema (str): The database schema for context
        prior_query (str): The previous query for reference
        model (str): The OpenAI model to use for generation
        
    Returns:
        str: A natural language variation of the original query
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
    You are a helpful assistant that rephrases questions while keeping the same meaning.
    
    DATABASE SCHEMA:
    {schema}
    
    ORIGINAL QUESTION:
    {query}
    
    PREVIOUS QUESTION (for reference only):
    {prior_query}
    
    Create a natural language variation of the ORIGINAL QUESTION that:
    1. Maintains the exact same meaning and intent
    2. Uses different wording and sentence structure
    3. Is clear and concise
    4. Is appropriate for the given database schema
    
    Return ONLY the rephrased question, with no additional text or explanation.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rephrases questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Slightly higher temperature for more creative variations
            max_tokens=100
        )
        
        # Extract and clean the response
        variation = response.choices[0].message.content.strip()
        
        # Remove any quotation marks if present
        variation = variation.strip('"\'')
        
        return variation if variation else query  # Fallback to original if empty
        
    except Exception as e:
        print(f"Error generating query variation: {str(e)}")
        return query  # Return original query on error



def generate_cypher_variation(
    query: str,
    schema: str,
    prior_query: str,
    model: str = "gpt-4"
) -> str:
    """
    Generates a variation of a Cypher query based on a new query and prior query.
    
    Args:
        query (str): The new query or question to answer
        schema (str): The database schema in a readable format
        prior_query (str): The previous Cypher query to use as a reference
        model (str): The OpenAI model to use for generation
        
    Returns:
        str: A new Cypher query that answers the new query while following
             a similar pattern to the prior query
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    prompt = f"""
    You are a Cypher query expert. Your task is to create a new Cypher query that answers
    a new question, using a prior query as a reference for style and structure.
    
    DATABASE SCHEMA:
    {schema}
    
    ORIGINAL QUESTION:
    {prior_query}
    
    PRIOR CYPHER QUERY:
    {prior_query}
    
    NEW QUESTION:
    {query}
    
    Create a new Cypher query that answers the NEW QUESTION while following a similar
    structure and style as the PRIOR CYPHER QUERY. Focus on:
    1. Maintaining similar query structure (MATCH, WHERE, RETURN clauses)
    2. Using similar variable naming conventions
    3. Following the same level of complexity
    4. Ensuring the query is valid for the given schema
    
    Return only the Cypher query, with no additional text or explanation.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Cypher queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more predictable results
            max_tokens=500
        )
        
        # Extract just the Cypher query from the response
        cypher_query = response.choices[0].message.content.strip()
        
        # Clean up the response to ensure it's just the query
        if '```' in cypher_query:
            # Remove code block markers if present
            cypher_query = cypher_query.split('```cypher')[-1].split('```')[0].strip()
            cypher_query = cypher_query.split('```')[-1].strip()
            
        return cypher_query
        
    except Exception as e:
        print(f"Error generating Cypher variation: {str(e)}")
        # Fallback: return the original query if generation fails
        return prior_query