import base64
import functions_framework
from flask import jsonify, request as flask_request
import os
from simple_graph import query_graph

def check_auth(username, password):
    """Check if a username/password combination is valid."""
    expected_username = os.environ.get("AUTH_USERNAME")
    expected_password = os.environ.get("AUTH_PASSWORD")
    return username == expected_username and password == expected_password

def authenticate():
    """Send a 401 response that enables basic auth."""
    return jsonify({
        "error": "Authentication required",
        "message": "Please provide valid credentials"
    }), 401, {
        'WWW-Authenticate': 'Basic realm="Authentication Required"'
    }

def requires_auth(f):
    """Decorator to require authentication on a route."""
    def decorated(*args, **kwargs):
        auth = flask_request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@functions_framework.http
@requires_auth
def query_graph_http(request):
    """HTTP Cloud Function that wraps the query_graph function.
    
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>

    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    # Parse request data
    request_json = request.get_json(silent=True)
    
    # Get query parameter
    query = None
    if request_json and 'query' in request_json:
        query = request_json['query']
    elif request.args and 'query' in request.args:
        query = request.args.get('query')
    
    if not query:
        return jsonify({"error": "No query provided. Please provide a 'query' parameter."}), 400
    
    # Get Neo4j credentials from environment variables
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_username = os.environ.get("NEO4J_USERNAME")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        return jsonify({"error": "Missing Neo4j credentials. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables."}), 500
    
    # Set default models (can be overridden via environment variables)
    retriever_model = os.environ.get("RETRIEVER_MODEL", "gpt-3.5-turbo")
    llm_model = os.environ.get("LLM_MODEL", "gpt-4")
    
    try:
        # Call the query_graph function
        result = query_graph(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            retriever_model=retriever_model,
            llm_model=llm_model,
            query=query
        )
        
        return {
            "answer": result.answer,
            "results": result.retriever_result
        }
        
    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error processing query: {str(e)}")
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500

# For local testing
if __name__ == "__main__":
    from flask import Flask, request, Response
    
    # Create a test app
    app = Flask(__name__)
    
    # Register the function with the test app
    @app.route("/", methods=["GET", "POST"])
    @requires_auth
    def test():
        return query_graph_http(flask_request)
    
    # Run the test server
    app.run(host="0.0.0.0", port=8080, debug=True)
