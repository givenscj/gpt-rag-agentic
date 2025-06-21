import asyncio
import logging

from typing import Dict, Any, List
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
        self.cosmos_container = self.cosmos_db.get_container_client(self.cosmos_container_name)
        self.partition_key = PartitionKey(path=self.cosmos_container_partition_key)

        self.aoai = AzureOpenAIClient(self.config)

    async def create_store(self):
        # Create the Cosmos DB store if it does not exist
        try:
            indexing_policy = {

            }

            vector_embed_policy = {

            }

            self.cosmos_db.create_container(
                id=self.cosmos_container_name,
                partition_key=self.partition_key,
                offer_throughput=1000,
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embed_policy
            )
            logging.info(f"[vector_index_retrieve] Cosmos DB container '{self.cosmos_container_name}' created successfully.")
        except Exception as e:
            error_message = f"Error creating Cosmos DB container: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
            return {"error": error_message}

    async def add_vector(self, vector : Dict):
        # Add a vector to the Cosmos DB store
        if not isinstance(vector, dict):
            error_message = "The vector must be a dictionary."
            logging.error(f"[vector_index_retrieve] {error_message}")
            return {"error": error_message}
        
        if 'id' not in vector:
            error_message = "The vector must contain an 'id' field."
            logging.error(f"[vector_index_retrieve] {error_message}")
            return {"error": error_message}
        
        try:
            self.cosmos_container.upsert_item(vector)
            logging.info(f"[vector_index_retrieve] Vector with ID {vector['id']} added successfully.")

        except Exception as e:
            error_message = f"Error adding vector: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
            return {"error": error_message}

    async def add_vectors(self, vectors: List[Dict]):
        # Add a vector to the Cosmos DB store via batch operation
        if not isinstance(vectors, list):
            error_message = "The vectors must be a list of dictionaries."
            logging.error(f"[vector_index_retrieve] {error_message}")
            return {"error": error_message}
        
        if not all(isinstance(vector, dict) for vector in vectors):
            error_message = "All items in the vectors list must be dictionaries."
            logging.error(f"[vector_index_retrieve] {error_message}")
            return {"error": error_message}
        
        if not all('id' in vector for vector in vectors):
            error_message = "Each vector must contain an 'id' field."
            logging.error(f"[vector_index_retrieve] {error_message}")
            return {"error": error_message}
        
        try:
            for vector in vectors:
                self.cosmos_container.upsert_item(vector, partition_key=self.partition_key)
            logging.info(f"[vector_index_retrieve] {len(vectors)} vectors added successfully.")
        except Exception as e:
            error_message = f"Error adding vectors: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
            return {"error": error_message}

    async def get_item(self, item_id: str) -> Dict:
        try:
            item = self.cosmos_container.read_item(item=item_id, partition_key=self.partition_key)
            return item
        except Exception as e:
            error_message = f"Error retrieving item with ID {item_id}: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
            return {"error": error_message}
        
    async def get_items(self, batch_size = 100) -> List[Dict]:
        items = []
        try:
            query = "SELECT * FROM c"
            container_items = self.cosmos_container.query_items(
                query=query,
                #partition_key=self.partition_key,
                max_item_count=batch_size,
                enable_cross_partition_query=True
            )

            continuation_token = None  # or stored token to resume
            pager = container_items.by_page(continuation_token)
            
            for page in pager:
                
                for item in page:
                    items.append(item)
                
                continuation_token = pager.continuation_token

        except Exception as e:
            error_message = f"Error retrieving all items: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
        
        return items

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
