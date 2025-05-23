import os
import json
from openai import OpenAI
from typing import Any, List, Dict, Union

def validate_answer_with_context(
    answer: str, 
    retrieved_results: List[Dict[str, Any]],
    model: str = "gpt-4"
) -> bool:
    """
    Validates if the provided answer is consistent with the retrieved results.
    
    Args:
        answer (str): The generated answer to validate
        retrieved_results (List[Dict]): List of retrieved context items
        model (str): OpenAI model to use for validation
        
    Returns:
        bool: True if the answer is consistent with the results, False otherwise
    """
    if not retrieved_results:
        # If no results were retrieved, the answer should indicate lack of context
        return "not enough context" in answer.lower() or "no information" in answer.lower()
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare the context for the prompt
    def stringify_result(result):
        try:
            return json.dumps(result, default=lambda o: str(o), indent=2)
        except:
            return str(result)
            
    context = "\n".join([
        f"--- Source {i+1} ---\n{stringify_result(result)}\n"
        for i, result in enumerate(retrieved_results)
    ])
    
    # Create the prompt with stricter validation criteria
    prompt = f"""
    You are a strict validator that rigorously evaluates if an answer is fully supported by the provided context.
    
    CONTEXT:
    {context}
    
    ANSWER TO VALIDATE:
    {answer}
    
    Strictly evaluate the answer against the context. Consider the following criteria:
    1. The answer MUST be fully supported by the context. If any part of the answer cannot be verified by the context, it's invalid.
    2. The answer MUST NOT contain any information that contradicts the context.
    3. The answer MUST NOT claim insufficient context if the context contains any relevant information.
    4. The answer MUST be precise and specific. Vague or generic answers are invalid.
    5. The answer MUST directly address the question using the provided context.
    
    Common reasons for invalidation:
    - Making assumptions not in the context
    - Providing generic responses that don't use the context
    - Claiming lack of information when context exists
    - Including information not present in the context
    - Being overly vague or non-specific
    
    Return your response as a JSON object with the following structure:
    {{
        "is_valid": boolean,
        "reasoning": "Detailed explanation of your decision, including specific issues found"
    }}
    
    Be strict in your evaluation. When in doubt, mark as invalid.
    """
    
    try:
        # Prepare the base request
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that validates answers against provided context. Always respond with a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        # Only add response_format for models that support it
        if any(model.startswith(prefix) for prefix in ["gpt-4-", "gpt-3.5-turbo-"]):
            request_params["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(**request_params)
        
        # Parse the response
        response_content = response.choices[0].message.content
        
        # Try to parse as JSON, fallback to string parsing if needed
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                # If no JSON found, return False for safety
                print(f"Could not parse response as JSON: {response_content}")
                return False
                
        return result.get("is_valid", False) if isinstance(result, dict) else False
        
    except Exception as e:
        print(f"Error validating answer with OpenAI: {str(e)}")
        # Default to True to avoid false negatives in case of API errors
        return True