import asyncio
import logging

from typing import Dict, Any
from connectors import CosmosDBClient
from azure.cosmos import PartitionKey
from .vector_store import VectorStore
from .types import VectorIndexRetrievalResult
from connectors import AzureOpenAIClient

from configuration import Configuration
from dependencies import get_config

class CosmosVectorStore(VectorStore):

    def __init__(self, config: Configuration = None):
        super().__init__()

        self.config = config

        if self.config is None:
            self.config = get_config()
        
        # Initialize the Cosmos DB vector store
        self.cosmos_client = CosmosDBClient(config)
        self.cosmos_db_name = config.get_value('COSMOS_DB_NAME', 'vectorstore')
        self.cosmos_container_name = config.get_value('COSMOS_CONTAINER_NAME', 'ragindex')
        self.cosmos_container_partition_key = config.get_value('COSMOS_CONTAINER_PARTITION_KEY', '/id')

        self.cosmos_db = self.cosmos_client.get_database(self.cosmos_db_name)
        self.cosmos_container = self.cosmos_db.get_container(self.cosmos_container_name)
        self.partition_key = PartitionKey(path=self.cosmos_container_partition_key)

        self.aoai = AzureOpenAIClient(self.config)

    async def add_vector(self, vector: list):
        # Add a vector to the Cosmos DB store
        pass

    async def query_vector(self, search_query: list):
        
        embeddings_query = await asyncio.to_thread(self.aoai.get_embeddings, search_query)

        cosmos_query = f"""SELECT TOP 10 c.title, VectorDistance(c.contentVector, {embeddings_query}) AS SimilarityScore
            FROM c
            ORDER BY VectorDistance(c.contentVector, {embeddings_query}) DESC"""

        results = []
        try:
            items = self.cosmos_container.query_items(
                query=cosmos_query,
                partition_key=self.partition_key
            )

            async for item in items:
                results.append({
                    "title": item.get("title"),
                    "similarity_score": item.get("SimilarityScore")
                })

        except Exception as e:
            error_message = f"Error querying Cosmos DB: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
        
        # Join the retrieved results into a single string.
        sources = ' '.join(result.get("title") for result in results)
        return VectorIndexRetrievalResult(result=sources, error=error_message)
