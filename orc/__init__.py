# function_app.py
import asyncio
import os
import json
import logging
import warnings
import azure.functions as func
from orchestration import RequestResponseOrchestrator, OrchestratorConfig
from configuration import Configuration

from configuration import Configuration
from dependencies import get_config
config :Configuration = get_config()

# Create the Function App with the desired auth level.
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

#@app.route(route="orc", methods=[func.HttpMethod.POST])
async def main(req: func.HttpRequest) -> func.HttpResponse:
    data = req.get_json()
    conversation_id = data.get("conversation_id")
    question = data.get("question")

    # Gather client principal info (optional)
    client_principal = {
        "id": data.get("client_principal_id", "00000000-0000-0000-0000-000000000000"),
        "name": data.get("client_principal_name", "anonymous"),
        "group_names": data.get("client_group_names", "")
    }
    access_token = data.get("access_token", None)
    
    if question:
        try:
            orchestrator = RequestResponseOrchestrator(conversation_id, OrchestratorConfig(), client_principal, access_token)
            result = await orchestrator.answer(question)
        except Exception as e:
            logging.error(f"Error processing request: {e}")

            while isinstance(e, Exception) and hasattr(e, 'exceptions') and e.exceptions:
                e = e.exceptions[0]
                logging.error(f"Exception details: {e.exceptions[0].exceptions[0]} - {str(e)}")
    
            return func.HttpResponse(body=json.dumps({"error": str(e)}), status_code=500)
        
        return func.HttpResponse(body=json.dumps(result))
    else:
        return func.HttpResponse(body={"error": "no question found in json input"}, status_code=400)