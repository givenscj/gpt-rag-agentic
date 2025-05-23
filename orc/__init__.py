# function_app.py
import asyncio
import os
import json
import logging
import warnings
import azure.functions as func
from orchestration import RequestResponseOrchestrator, OrchestratorConfig
from configuration import Configuration

config = Configuration()

# User Warning configuration
import warnings
# Available options for USER_WARNING_FILTER:
#   ignore  - never show the warning
#   always  - always show the warning
#   error   - turn the warning into an exception
#   once    - show the warning only once
#   module  - show the warning only once per module
#   default - default Python behavior
user_warning_filter = config.get_value('USER_WARNING_FILTER', 'ignore').lower()
warnings.filterwarnings(user_warning_filter, category=UserWarning)

# Logging configuration
logging.basicConfig(level=config.get_value('LOGLEVEL', 'INFO').upper(), force=True)
logging.getLogger("azure").setLevel(config.get_value('AZURE_LOGLEVEL', 'WARNING').upper())
logging.getLogger("httpx").setLevel(config.get_value('HTTPX_LOGLEVEL', 'ERROR').upper())
logging.getLogger("httpcore").setLevel(config.get_value('HTTPCORE_LOGLEVEL', 'ERROR').upper())
logging.getLogger("openai._base_client").setLevel(config.get_value('OPENAI_BASE_CLIENT_LOGLEVEL', 'WARNING').upper())
logging.getLogger("urllib3").setLevel(config.get_value('URLLIB3_LOGLEVEL', 'WARNING').upper())
logging.getLogger("urllib3.connectionpool").setLevel(config.get_value('URLLIB3_CONNECTIONPOOL_LOGLEVEL', 'WARNING').upper())
logging.getLogger("openai").setLevel(config.get_value('OPENAI_LOGLEVEL', 'WARNING').upper())
logging.getLogger("autogen_core").setLevel(config.get_value('AUTOGEN_CORE_LOGLEVEL', 'WARNING').upper())
logging.getLogger("autogen_core.events").setLevel(config.get_value('AUTOGEN_EVENTS_LOGLEVEL', 'WARNING').upper())
logging.getLogger("uvicorn.error").propagate = True
logging.getLogger("uvicorn.access").propagate = True

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
        orchestrator = RequestResponseOrchestrator(conversation_id, OrchestratorConfig(), client_principal, access_token)
        result = await orchestrator.answer(question)
        return func.HttpResponse(body=json.dumps(result))
    else:
        return func.HttpResponse(body={"error": "no question found in json input"}, status_code=400)