# function_app.py
import asyncio
import os
import json
import logging
import warnings
import azure.functions as func
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, JSONResponse
from orchestration import RequestResponseOrchestrator, StreamingOrchestrator, OrchestratorConfig

from configuration import Configuration
from dependencies import get_config
config :Configuration = get_config()

# Create the Function App with the desired auth level.
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

#@app.route(route="orcstream", methods=[func.HttpMethod.POST])
async def main(req: func.HttpRequest) -> StreamingResponse:
    data = await req.get_json()
    conversation_id = data.get("conversation_id")
    question = data.get("question")
    optimize_for_audio = data.get("optimize_for_audio", False)    

    # Gather client principal info (optional)
    client_principal = {
        "id": data.get("client_principal_id", "00000000-0000-0000-0000-000000000000"),
        "name": data.get("client_principal_name", "anonymous"),
        "group_names": data.get("client_group_names", "")
    }
    access_token = data.get("access_token", None)
    
    if question:
        orchestrator = StreamingOrchestrator(conversation_id, OrchestratorConfig(), client_principal, access_token)
        orchestrator.set_optimize_for_audio(optimize_for_audio)

        async def stream_generator():
            logging.info("[orcstream_endpoint] Entering stream_generator")
            last_yield = asyncio.get_event_loop().time()
            heartbeat_interval = 15  # seconds between heartbeats
            heartbeat_count = 0

            async for chunk in orchestrator.answer(question):
                now = asyncio.get_event_loop().time()
                # If the time since the last yield exceeds the heartbeat interval, send a heartbeat
                if now - last_yield >= heartbeat_interval:
                    heartbeat_count += 1
                    logging.info(f"Sending heartbeat #{heartbeat_count}")
                    yield "\n\n"
                    last_yield = now
                if chunk:
                    # logging.info(f"Yielding chunk: {chunk}")
                    # For text-only mode, yield the raw chunk; else, serialize to JSON.
                    yield chunk
                    last_yield = now
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        return JSONResponse(content={"error": "no question found in json input"}, status_code=400)