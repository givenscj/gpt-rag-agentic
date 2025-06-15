import re
import time
import json
import logging
import asyncio
from typing import Annotated, Optional, List, Dict, Any
from urllib.parse import urlparse

from connectors import AzureOpenAIClient

from .types import (
    VectorIndexRetrievalResult,
    MultimodalVectorIndexRetrievalResult,
    DataPointsResult,
)

from factories.vector_stores.vector_store_factory import VectorStoreFactory

from configuration import Configuration
from dependencies import get_config
config :Configuration = get_config()

# -----------------------------------------------------------------------------
# Main Functions
# -----------------------------------------------------------------------------

async def vector_index_retrieve(
    input: Annotated[
        str, "An optimized query string based on the user's ask and conversation history, when available"
    ],
    security_ids: str = 'anonymous',
    vector_strategy: Annotated[str, "The vector search strategy to use, e.g., 'cosmos', 'postgres', 'elasticsearch', 'azure_sql', 'aisearch'] = 'cosmos'"] = 'aisearch'
) -> Annotated[
    VectorIndexRetrievalResult, "A Pydantic model containing the search results as a string"
]:
    """
    Performs a vector search against Azure Cognitive Search and returns the results
    wrapped in a Pydantic model. If an error occurs, the 'error' field is populated.
    """
    factory = VectorStoreFactory.get_factory(vector_strategy)
    return factory.query_vector(input, security_ids)