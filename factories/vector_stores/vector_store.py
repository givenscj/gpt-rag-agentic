import re
import json
import logging

from urllib.parse import urlparse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .types import VectorIndexRetrievalResult, MultimodalVectorIndexRetrievalResult, DataPointsResult

class VectorStore:

    supports_batch_migration = True
    max_batch_size = 1000
    max_workers = 10

    def __init__(self):
        # Initialize the vector store
        pass

    async def create_store(self):
        # Create the vector store
        pass

    async def add_vector(self, vector: Dict, fields: Optional[List[str]] = None):
        # Add a vector to the store
        pass

    async def add_vectors(self, vector: list):
        # Add a vector to the store
        pass

    async def get_fields(self):
        # Add a vector to the store
        pass

    async def get_item(self, item_id: str) -> Dict:
        pass

    async def get_items(self, batch_size = 100) -> List[Dict]:
        pass

    async def query_vector(self, query: list):
        # Query the vector store
        pass

    def extract_captions(str_captions):
        # Regular expression pattern to match image references followed by their descriptions
        pattern = r"\[.*?\]:\s(.*?)(?=\[.*?\]:|$)"
        
        # Find all matches
        matches = re.findall(pattern, str_captions, re.DOTALL)

        return [match.strip() for match in matches]

    def replace_image_filenames_with_urls(content: str, related_images: list) -> str:
        """
        Replace image filenames or relative paths in the content string with their corresponding full URLs
        from the related_images list.
        """
        for image_url in related_images:
            # Parse the URL and remove the leading slash from the path
            logging.debug(f"[multimodal_vector_index_retrieve] image_url: {image_url}.")
            parsed_url = urlparse(image_url)
            image_path = parsed_url.path.lstrip('/')  # e.g., 'documents-images/myfolder/filename.png'
            logging.debug(f"[multimodal_vector_index_retrieve] image_path: {image_path}.")
            # Replace occurrences of the relative path in the content with the full URL
            content = content.replace(image_path, image_url)
            logging.debug(f"[multimodal_vector_index_retrieve] content: {content}.")
            # Also replace only the filename if it appears alone
            # filename = image_path.split('/')[-1]
            # content = content.replace(filename, image_url)

        return content

    def get_data_points_from_chat_log(chat_log: list) -> DataPointsResult:
        """
        Parses a chat log to extract data points (e.g., filenames with extension) from tool call events.
        Returns a Pydantic model containing the list of extracted data points.
        """
        # Regex patterns.
        request_call_id_pattern = re.compile(r"id='([^']+)'")
        request_function_name_pattern = re.compile(r"name='([^']+)'")
        exec_call_id_pattern = re.compile(r"call_id='([^']+)'")
        exec_content_pattern = re.compile(r"content='(.+?)', call_id=", re.DOTALL)

        # Allowed file extensions.
        allowed_extensions = ['vtt', 'xlsx', 'xls', 'pdf', 'docx', 'pptx', 'png', 'jpeg', 'jpg', 'bmp', 'tiff']
        filename_pattern = re.compile(
            rf"([^\s:]+\.(?:{'|'.join(allowed_extensions)})\s*:\s*.*?)(?=[^\s:]+\.(?:{'|'.join(allowed_extensions)})\s*:|$)",
            re.IGNORECASE | re.DOTALL
        )

        relevant_call_ids = set()
        data_points = []

        for msg in chat_log:
            if msg["message_type"] == "ToolCallRequestEvent":
                content = msg["content"][0]
                call_id_match = request_call_id_pattern.search(content)
                function_name_match = request_function_name_pattern.search(content)
                if call_id_match and function_name_match:
                    if function_name_match.group(1) == "vector_index_retrieve_wrapper":
                        relevant_call_ids.add(call_id_match.group(1))
            elif msg["message_type"] == "ToolCallExecutionEvent":
                content = msg["content"][0]
                call_id_match = exec_call_id_pattern.search(content)
                if call_id_match and call_id_match.group(1) in relevant_call_ids:
                    content_part_match = exec_content_pattern.search(content)
                    if not content_part_match:
                        continue
                    content_part = content_part_match.group(1)
                    try:
                        parsed = json.loads(content_part)
                        texts = parsed.get("texts", [])
                    except json.JSONDecodeError:
                        texts = [re.split(r'["\']images["\']\s*:\s*\[', content_part, 1, re.IGNORECASE)[0]]
                    for text in texts:
                        text = bytes(text, "utf-8").decode("unicode_escape")
                        for match in filename_pattern.findall(text):
                            extracted = match.strip(" ,\\\"").lstrip("[").rstrip("],")
                            if extracted:
                                data_points.append(extracted)
        return DataPointsResult(data_points=data_points)

class VectorStoreMigration:

    source_store: VectorStore
    target_store: VectorStore
    batch_size: int = 100
    max_workers: int = 10

    def __init__(self, source_store: VectorStore, target_store: VectorStore, batch_size: int = 100, max_workers: int = 10):
        self.source_store = source_store
        self.target_store = target_store
        self.batch_size = batch_size
        self.max_workers = max_workers

    async def migrate(self):
        # Migrate vectors from source to target store
        fields = await self.target_store.get_fields()
        items = await self.source_store.get_items()
        for item in items:
            try:
                await self.target_store.add_vector(item, fields=fields)
            except Exception as e:
                error_message = f"Error migrating item {item.get('id', 'unknown')}: {str(e)}"
                logging.error(f"[vector_store_migration] {error_message}", exc_info=True)
                continue

        logging.info(f"[vector_store_migration] Migrated {len(items)} items from {self.source_store} to {self.target_store}")        

    async def migrate_batch(self, batch_size: Optional[int] = None):
        # Migrate vectors from source to target store
        if batch_size is None:
            batch_size = self.batch_size

        while True:
            vectors = await self.source_store.get_items(batch_size=batch_size)
            
            if not vectors:
                break
            
            await self.target_store.add_vectors(vectors)
