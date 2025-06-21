import logging
import asyncio
import aiohttp
import time
import re

from configuration  import Configuration
from connectors import AzureOpenAIClient
from typing import List, Dict, Any, Optional, Annotated
from .vector_store import VectorStore
from .types import VectorIndexRetrievalResult, MultimodalVectorIndexRetrievalResult

VECTOR_SEARCH_APPROACH = 'vector'
TERM_SEARCH_APPROACH = 'term'
HYBRID_SEARCH_APPROACH = 'hybrid'

class AISearchVectorStore(VectorStore):

    def __init__(self, config: Configuration = None):
        super().__init__()
        
        self.config = config or Configuration()
        self.aoai = AzureOpenAIClient(self.config)

        self.search_service = self.config.get_value('AZURE_SEARCH_SERVICE')
        self.search_index = self.config.get_value('AZURE_SEARCH_INDEX', 'ragindex')
        self.search_api_version = self.config.get_value('AZURE_SEARCH_API_VERSION', '2024-07-01')

        self.search_top_k = self.config.get_value('AZURE_SEARCH_TOP_K', 3)
        self.search_approach = self.config.get_value('AZURE_SEARCH_APPROACH', 'hybrid')
        self.semantic_search_config = self.config.get_value('AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG', 'my-semantic-config')
        self.use_semantic = self.config.get_value('AZURE_SEARCH_USE_SEMANTIC', 'false').lower() == 'true'

        self.search_url = (
            f"https://{self.search_service}.search.windows.net/indexes/{self.search_index}/docs"
            f"?api-version={self.search_api_version}"
        )

    async def create_store(self):
        """
        Creates the Azure Search index if it does not already exist.
        """
        try:
            # Acquire token for Azure Search.
            azure_search_token = await self._get_azure_search_token()

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {azure_search_token}'
            }

            search_url = (
                f"https://{self.search_service}.search.windows.net/indexes/{self.search_index}?api-version={self.search_api_version}"
            )

            # Check if the index already exists.
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        logging.info(f"[vector_index_retrieve] Index '{self.search_index}' already exists.")
                        return

            # If the index does not exist, create it.
            index_definition = {
                "name": self.search_index,
                "fields": [
                    {"name": "id", "type": "Edm.String", "key": True, "searchable": False},
                    {"name": "content", "type": "Edm.String", "searchable": True, "analyzer": "standard.lucene"},
                    {"name": "contentVector", "type": "Collection(Edm.Single)", "searchable": False, "vectorSearchDimensions": 1536},
                    {"name": "title", "type": "Edm.String", "searchable": True, "analyzer": "standard.lucene"},
                    {"name": "url", "type": "Edm.String", "searchable": False},
                    {"name": "filepath", "type": "Edm.String", "searchable": False},
                    {"name": "chunk_id", "type": "Edm.String", "searchable": False},
                    {"name": "metadata_security_id", "type": "Collection(Edm.String)", "searchable": True}
                ],
                "@odata.type": "#Microsoft.Azure.Search.Index"
            }

            async with aiohttp.ClientSession() as session:
                async with session.put(search_url, headers=headers, json=index_definition) as response:
                    if response.status == 201:
                        logging.info(f"[vector_index_retrieve] Successfully created index '{self.search_index}'.")
                    else:
                        text = await response.text()
                        error_message = f"Error creating index: {response.status} - {response.reason} - {text}"
                        logging.error(f"[vector_index_retrieve] {error_message}")
                        raise Exception(error_message)
        except Exception as e:  
            error_message = f"Error creating Azure Search index: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
            raise Exception(error_message) from e


    async def _get_azure_search_token(self) -> str:
        """
        Acquires an Azure Search access token using chained credentials.
        """
        try:
            # Wrap the synchronous token acquisition in a thread.
            token_obj = await asyncio.to_thread(self.config.credential.get_token, "https://search.azure.com/.default")
            return token_obj.token
        except Exception as e:
            logging.error("Error obtaining Azure Search token.", exc_info=True)
            raise Exception("Failed to obtain Azure Search token.") from e

    async def _perform_search(self, url: str, headers: Dict[str, str], body: Dict[str, Any], method : str='POST') -> Dict[str, Any]:
        """
        Performs an asynchronous HTTP POST request to the given URL with the provided headers and body.
        Returns the parsed JSON response.

        Raises:
            Exception: When the request fails or returns an error status.
        """
        async with aiohttp.ClientSession() as session:
            try:
                if method.upper() == 'GET':
                    async with session.get(url, headers=headers, params=body) as response:
                        if response.status >= 400:
                            text = await response.text()
                            error_message = f"Error {response.status}: {text}"
                            logging.error(f"[_perform_search] {error_message}")
                            raise Exception(error_message)
                        return await response.json()
                elif method.upper() == 'POST':
                    async with session.post(url, headers=headers, json=body) as response:
                        if response.status >= 400:
                            text = await response.text()
                            error_message = f"Error {response.status}: {text}"
                            logging.error(f"[_perform_search] {error_message}")
                            raise Exception(error_message)
                        return await response.json()
                elif method.upper() == 'PUT':
                    async with session.put(url, headers=headers, json=body) as response:
                        if response.status >= 400:
                            text = await response.text()
                            error_message = f"Error {response.status}: {text}"
                            logging.error(f"[_perform_search] {error_message}")
                            raise Exception(error_message)
                        return await response.json()
            except Exception as e:
                logging.error("Error during asynchronous HTTP request.", exc_info=True)
                raise Exception("Failed to execute search query.") from e

    async def get_fields(self) -> List[str]:
        """
        Retrieves the fields of the Azure Search index.
        Returns a list of field names.
        """
        try:
            # Acquire token for Azure Search.
            azure_search_token = await self._get_azure_search_token()

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {azure_search_token}'
            }

            search_url = (
                f"https://{self.search_service}.search.windows.net/indexes/{self.search_index}?api-version={self.search_api_version}"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status >= 400:
                        text = await response.text()
                        error_message = f"Error {response.status}: {text}"
                        logging.error(f"[vector_index_retrieve] {error_message}")
                        raise Exception(error_message)
                    index_info = await response.json()
                    return [field['name'] for field in index_info.get('fields', [])]
        except Exception as e:
            error_message = f"Error retrieving fields: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
            raise Exception(error_message) from e
    
    async def add_vector(self, vector : Dict, fields: Optional[List[str]] = None):
        # Add a vector to the Azure AI store

        #remove any fields not in the index
        if not isinstance(vector, dict):
            error_message = "The vector must be a dictionary."
            logging.error(f"[vector_index_retrieve] {error_message}")
            raise Exception(error_message)
        
        if 'id' not in vector:
            error_message = "The vector must contain an 'id' field."
            logging.error(f"[vector_index_retrieve] {error_message}")
            raise Exception(error_message)
        
        if fields is None:
            # If no fields are provided, retrieve the index fields.
            fields = await self.get_fields()

        # Filter the vector to only include fields that are in the index.
        vector = {k: v for k, v in vector.items() if k in fields}
        
        # create the body for the vector
        body = {
            "value": [vector],
            "@search.action": "upload"
        }

        try:
            # Acquire token for Azure Search.
            azure_search_token = await self._get_azure_search_token()

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {azure_search_token}'
            }

            search_url = (
                f"https://{self.search_service}.search.windows.net/indexes/{self.search_index}/docs/index"
                f"?api-version={self.search_api_version}"
            )

            # Perform the search asynchronously.
            response_json = await self._perform_search(search_url, headers, body)

            if response_json.get('value'):
                logging.info(f"[vector_index_retrieve] Successfully added vector: {vector}")
            else:
                logging.warning("[vector_index_retrieve] No response value returned after adding vector")
        except Exception as e:
            error_message = f"Error adding vector: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
            raise Exception(error_message) from e

    async def add_vectors(self, vectors: List[Dict]):
        # Add a set of vector to the Azure Ai store via batch upload.
        if not vectors:
            logging.warning("[vector_index_retrieve] No vectors provided for batch upload.")
            return
        
        # Create the body for the batch upload.
        body = {
            "value": vectors,
            "@search.action": "upload"
        }

        try:
            # Acquire token for Azure Search.
            azure_search_token = await self._get_azure_search_token()

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {azure_search_token}'
            }

            # Perform the search asynchronously.
            response_json = await self._perform_search(self.search_url, headers, body)

            if response_json.get('value'):
                logging.info(f"[vector_index_retrieve] Successfully added {len(vectors)} vectors")
            else:
                logging.warning("[vector_index_retrieve] No response value returned after batch upload")
        except Exception as e:
            error_message = f"Error adding vectors: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
            raise Exception(error_message) from e

    async def get_items(self, batch_size = 100) -> list:
        items = []
        try:
            # Acquire token for Azure Search.
            azure_search_token = await self._get_azure_search_token()

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {azure_search_token}'
            }

            body = {
                "search": "*",
                "top": batch_size,
                "skip": 0,  # Start from the beginning
                "count": 'true'  # Include total count in the response
            }

            search_url = (
                f"https://{self.search_service}.search.windows.net/indexes/{self.search_index}/docs/search"
                f"?api-version={self.search_api_version}"
            )

            while True:
                # Execute the search query asynchronously.
                response_json = await self._perform_search(search_url, headers, body, method='POST')

                if 'value' in response_json:
                    items.extend(response_json['value'])
                    logging.info(f"[vector_index_retrieve] Retrieved {len(response_json['value'])} items from Azure Search")
                else:
                    logging.info("[vector_index_retrieve] No items found in Azure Search")
                    break

                # Check if we have more items to fetch.
                if len(items) >= response_json.get('@odata.count', 0):
                    break

                # Update the body for the next batch.
                body['skip'] += batch_size

        except Exception as e:
            error_message = f"Error retrieving all items: {str(e)}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)
        
        return items

    async def query_vector(self, search_query: list, security_ids: str = '') -> VectorIndexRetrievalResult:

        search_results: List[str] = []
        error_message: Optional[str] = None
        
        try:
            # Generate embeddings for the query.
            start_time = time.time()
            logging.info(f"[vector_index_retrieve] Generating question embeddings. Search query: {search_query}")
            # Wrap synchronous get_embeddings in a thread.
            embeddings_query = await asyncio.to_thread(self.aoai.get_embeddings, search_query)
            response_time = round(time.time() - start_time, 2)
            logging.info(f"[vector_index_retrieve] Finished generating embeddings in {response_time} seconds")

            # Acquire token for Azure Search.
            azure_search_token = await self._get_azure_search_token()

            # Prepare the request body.
            body: Dict[str, Any] = {
                "select": "title, content, url, filepath, chunk_id",
                "top": self.search_top_k
            }
            if self.search_approach == TERM_SEARCH_APPROACH:
                body["search"] = search_query
            elif self.search_approach == VECTOR_SEARCH_APPROACH:
                body["vectorQueries"] = [{
                    "kind": "vector",
                    "vector": embeddings_query,
                    "fields": "contentVector",
                    "k": int(self.search_top_k)
                }]
            elif self.search_approach == HYBRID_SEARCH_APPROACH:
                body["search"] = search_query
                body["vectorQueries"] = [{
                    "kind": "vector",
                    "vector": embeddings_query,
                    "fields": "contentVector",
                    "k": int(self.search_top_k)
                }]

            # Apply security filter.
            filter_str = (
                f"metadata_security_id/any(g:search.in(g, '{security_ids}')) "
                f"or not metadata_security_id/any()"
            )
            body["filter"] = filter_str
            logging.debug(f"[vector_index_retrieve] Search filter: {filter_str}")

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {azure_search_token}'
            }

            # Execute the search query asynchronously.
            start_time = time.time()
            response_json = await self._perform_search(self.search_url, headers, body)
            elapsed = round(time.time() - start_time, 2)
            logging.info(f"[vector_index_retrieve] Finished querying Azure Cognitive Search in {elapsed} seconds")

            if response_json.get('value'):
                logging.info(f"[vector_index_retrieve] {len(response_json['value'])} documents retrieved")
                for doc in response_json['value']:
                    url = doc.get('url', '')
                    uri = re.sub(r'https://[^/]+\.blob\.core\.windows\.net', '', url)                
                    content_str = doc.get('content', '').strip()
                    search_results.append(f"{uri}: {content_str}\n")
            else:
                logging.info("[vector_index_retrieve] No documents retrieved")

        except Exception as e:
            error_message = f"Exception occurred: {e}"
            logging.error(f"[vector_index_retrieve] {error_message}", exc_info=True)

        # Join the retrieved results into a single string.
        sources = ' '.join(search_results)
        return VectorIndexRetrievalResult(result=sources, error=error_message)
    
    async def multimodal_vector_index_retrieve(self,
        input: Annotated[
            str, "An optimized query string based on the user's ask and conversation history, when available"
        ],
        security_ids: str = 'anonymous'
    ) -> Annotated[
        MultimodalVectorIndexRetrievalResult,
        "A Pydantic model containing the search results with separate lists for texts and images"
    ]:
        """
        Variation of vector_index_retrieve that fetches text and related images from the search index.
        Returns the results wrapped in a Pydantic model with separate lists for texts and images.
        """

        logging.info(f"[multimodal_vector_index_retrieve] User input: {input}")

        text_results: List[str] = []
        image_urls: List[List[str]] = []
        captions: List[str] = []
        error_message: Optional[str] = None

        # 1. Generate embeddings for the query.
        try:
            start_time = time.time()
            embeddings_query = await asyncio.to_thread(self.aoai.get_embeddings, input)
            embedding_time = round(time.time() - start_time, 2)
            logging.info(f"[multimodal_vector_index_retrieve] Query embeddings took {embedding_time} seconds")
        except Exception as e:
            error_message = f"Error generating embeddings: {e}"
            logging.error(f"[multimodal_vector_index_retrieve] {error_message}", exc_info=True)
            return MultimodalVectorIndexRetrievalResult(
                texts=[],
                images=[],
                error=error_message
            )

        # 2. Acquire Azure Search token.
        try:
            azure_search_token = await self._get_azure_search_token()
        except Exception as e:
            error_message = f"Error acquiring token for Azure Search: {e}"
            logging.error(f"[multimodal_vector_index_retrieve] {error_message}", exc_info=True)
            return MultimodalVectorIndexRetrievalResult(
                texts=[],
                images=[],
                error=error_message
            )

        # 3. Build the request body.
        body: Dict[str, Any] = {
            "select": "title, content, filepath, url, imageCaptions, relatedImages",
            "top": self.search_top_k,
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": embeddings_query,
                    "fields": "contentVector",
                    "k": int(self.search_top_k)
                },
                {
                    "kind": "vector",
                    "vector": embeddings_query,
                    "fields": "captionVector",
                    "k": int(self.search_top_k)
                }
            ]
        }

        if self.use_semantic and self.search_approach != "vector":
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = self.semantic_search_config

        # Apply security filter.
        filter_str = (
            f"metadata_security_id/any(g:search.in(g, '{security_ids}')) "
            "or not metadata_security_id/any()"
        )
        body["filter"] = filter_str

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {azure_search_token}'
        }

        # 4. Query Azure Search.
        try:
            start_time = time.time()
            response_json = await self._perform_search(self.search_url, headers, body)
            response_time = round(time.time() - start_time, 2)
            logging.info(f"[multimodal_vector_index_retrieve] Finished querying Azure AI Search in {response_time} seconds")

            for doc in response_json.get('value', []):

                content = doc.get('content', '')
                str_captions = doc.get('imageCaptions', '')        
                captions.append(self.extract_captions(str_captions))
                url = doc.get('url', '')

                # Convert blob URL to relative path
                uri = re.sub(r'https://[^/]+\.blob\.core\.windows\.net', '', url)
                text_results.append(f"{uri}: {content.strip()}")

                # Replace image filenames with URLs
                content = self.replace_image_filenames_with_urls(content, doc.get('relatedImages', []))

                # Extract image URLs from <figure> tags
                # doc_image_urls = re.findall(r'<figure>(https?://.*?)</figure>', content)
                # image_urls.append(doc_image_urls)
                image_urls.append(doc.get('relatedImages', []))

                # Replace <figure>...</figure> with <img src="...">
                # content = re.sub(r'<figure>(https?://\S+)</figure>', r'<img src="\1">', content)

        except Exception as e:
            error_message = f"Exception in retrieval: {e}"
            logging.error(f"[multimodal_vector_index_retrieve] {error_message}", exc_info=True)

        return MultimodalVectorIndexRetrievalResult(
            texts=text_results,
            images=image_urls,
            captions=captions,
            error=error_message
        )