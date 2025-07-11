import base64
import json
import logging
import re
from typing import Annotated, Sequence

from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base._chat_agent import Response
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    MultiModalMessage,
    TextMessage,
    ToolCallSummaryMessage,
)
from autogen_core import CancellationToken, Image
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.tools import FunctionTool
from connectors import BlobClient
from tools import get_time, get_today_date
from factories.vector_stores.types import MultimodalVectorIndexRetrievalResult
from factories.vector_stores.constants import VectorStoreType
from factories.vector_stores.vector_store_factory import VectorStoreFactory

from .constants import Strategy
from .base_agent_strategy import BaseAgentStrategy
from dependencies import get_config
from configuration import Configuration

class MultimodalMessageCreator(BaseChatAgent):
    """
    A custom agent that constructs a MultiModalMessage from the vector_index_retrieve_wrapper 
    tool results. The actual tool call is done by another agent 
    (retrieval_agent) in the same SelectorGroupChat. 
    This agent simply scans the conversation for the tool's result, parses 
    text + image URLs, and returns a MultiModalMessage.
    """
    def __init__(self, name: str, system_prompt: str, model_context: BufferedChatCompletionContext, config :Configuration = None):
        super().__init__(
            name=name,
            description="An agent that creates `MultiModalMessage` objects from the results of `vector_index_retrieve_wrapper`, executed by an `AssistantAgent` called `retrieval_agent`."
        )
        self._last_multimodal_result = None
        self.system_prompt = system_prompt + "\n\n"
        self._model_context = model_context

        self.config = config or get_config()

        self.vector_store_type = self.config.get_value("VECTOR_STORE_STRATEGY", VectorStoreType.AISEARCH.value)

    @property
    def produced_message_types(self):
        """
        Return the message types this agent can produce.
        We produce a MultiModalMessage.
        """
        return (MultiModalMessage,)

    async def on_messages(
        self, 
        messages: Sequence[ChatMessage], 
        cancellation_token: CancellationToken
    ) -> Response:
        """
        Handles incoming messages to process vector index retrieval results 
        and construct a MultiModalMessage response.
        """
        retrieval_data = None

        # Iterate through messages in reverse to find the latest relevant tool output
        for msg in reversed(messages):
            if isinstance(msg, ToolCallSummaryMessage):
                try:
                    msg_content = msg.content   
                    parsed_content = json.loads(msg_content)
                    if "texts" in parsed_content or "images" in parsed_content:
                        retrieval_data = parsed_content
                        break
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse message content as JSON: {e}")
                    continue

        if not retrieval_data:
            # Fallback response when no relevant data is found
            fallback_msg = TextMessage(
                content="No vector retrieval data was found in the conversation.",
                source=self.name
            )
            return Response(chat_message=fallback_msg)

        # Extract text and image data
        texts = retrieval_data.get("texts", [])
        image_urls_list = retrieval_data.get("images", [])
        captions_lists = retrieval_data.get("captions", [])

        # Combine text snippets into a single string
        combined_text = self.system_prompt + "\n\n".join(texts) if texts else "No text results"
        logging.debug(f"[multimodal_agent_strategy] combined_text: {combined_text}")

        # Fetch images from URLs
        image_objects = []
        max_images = 50  # maximum number of images to process (Azure OpenaI GPT-4o limit)
        image_count = 0
        document_count = 0
        for image_urls_list in image_urls_list:  # Assuming each item in image_urls is a list of URLs
            image_count = 0
            for url in image_urls_list:  # Iterate through each URL in the sublist
                if image_count >= max_images:
                    logging.info(f"[multimodal_agent_strategy] Reached the maximum image limit of {max_images}. Stopping further downloads.")
                    break  # Stop processing more URLs                
                try:

                    # Initialize BlobClient with the blob URL
                    blob_client = BlobClient(blob_url=url)
                    logging.debug(f"[multimodal_agent_strategy] Initialized BlobClient for URL: {url}")
                    
                    # Download the blob data as bytes
                    blob_data = blob_client.download_blob()
                    logging.debug(f"[multimodal_agent_strategy] Downloaded blob data for URL: {url}")
                    
                    # Open the image using PIL
                    base64_str = base64.b64encode(blob_data).decode('utf-8')
                    pil_img = Image.from_base64(base64_str)
                    logging.debug(f"[multimodal_agent_strategy] Opened image from URL: {url}")
                    uri = re.sub(r'https://[^/]+\.blob\.core\.windows\.net', '', url)
                    pil_img.filepath = uri
                    logging.debug(f"[multimodal_agent_strategy] Filepath (uri): {uri}")
                    pil_img.caption = captions_lists[document_count][image_count] if image_count < len(image_urls_list) else None                 

                    # Append the PIL Image object to your list (modify as needed)
                    image_objects.append(pil_img)
                    image_count += 1  # Increment the counter
                    logging.info(f"[multimodal_agent_strategy] Successfully loaded image from {url}")

                except Exception as e:
                    logging.error(f"[multimodal_agent_strategy] Could not load image from {url}: {e}")
                image_count += 1  # Increment the image number for the next set of URLs
            document_count += 1  # Increment the document number for the next set of URLs

        # Construct and return the MultiModalMessage response
        multimodal_msg = MultiModalMessage(
            content=[combined_text, *image_objects],
            source=self.name
        )
        return Response(chat_message=multimodal_msg)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """
        Reset the agent state if needed. 
        In this basic example, we clear the internal variable.
        """
        self._last_multimodal_result = None

class MultimodalAgentStrategy(BaseAgentStrategy):
    def __init__(self):
        super().__init__()
        self.strategy_type = Strategy.MULTIMODAL_RAG

    async def create_agents(self, history, client_principal=None, access_token=None, output_mode=None, output_format=None):
        """
        Multimodal RAG creation strategy that creates the basic agents and registers functions.
        
        Parameters:
        - history: The conversation history, which will be summarized to provide context for the assistant's responses.
        
        Returns:
        - agent_configuration: A dictionary that includes the agents team, default model client, termination conditions and selector function.

        Note:
        To use a different model for an specific agent, instantiate a separate AzureOpenAIChatCompletionClient and assign it instead of using self._get_model_client().
        """

        # Model Context
        shared_context = await self._get_model_context(history) 

        # Wrapper Functions for Tools

        async def vector_index_retrieve_wrapper(
            input: Annotated[str, "An optimized query string based on the user's ask and conversation history, when available"],
            security_ids: str = 'anonymous'
        ) -> MultimodalVectorIndexRetrievalResult:
            factory = VectorStoreFactory.get_factory(self.vector_store_type)
            return factory.multimodal_vector_index_retrieve(input, security_ids)

        vector_index_retrieve_tool = FunctionTool(
            vector_index_retrieve_wrapper, name="vector_index_retrieve", description="Performs a vector search using Azure AI Search fetching text and related images get relevant sources for answering the user's query."
        )

        # Agents

        ## Triage Agent
        triage_prompt = await self._read_prompt("triage_agent")
        triage_agent = AssistantAgent(
            name="triage_agent",
            system_message=triage_prompt,
            model_client=self._get_model_client(), 
            tools=[vector_index_retrieve_tool],
            reflect_on_tool_use=False,
            model_context=shared_context
        )

        ## Multimodal Message Creator
        multimodal_rag_message_prompt = await self._read_prompt("multimodal_rag_message")
        multimodal_creator = MultimodalMessageCreator(name="multimodal_creator", system_prompt=multimodal_rag_message_prompt, model_context=shared_context)

        ## Assistant Agent
        main_assistant_prompt = await self._read_prompt("main_assistant")        
        main_assistant = AssistantAgent(
            name="main_assistant",
            system_message=main_assistant_prompt,
            model_client=self._get_model_client(),
            reflect_on_tool_use=True
        )
        
        ## Chat Closure Agent
        chat_closure = await self._create_chat_closure_agent(output_format, output_mode)

        # Agent Configuration

        # Optional: Override the termination condition for the assistant. Set None to disable each termination condition.
        # self.max_rounds = int(os.getenv('MAX_ROUNDS', 8))
        # self.terminate_message = "TERMINATE"

        def custom_selector_func(messages):
            """
            Selects the next agent based on the source of the last message.
            
            Transition Rules:
                user -> triage_agent
                triage_agent (ToolCallSummaryMessage) -> multimodal_creator
                multimodal_creator -> assistant
                Other -> None (SelectorGroupChat will handle transition)
            """            
            last_msg = messages[-1]
            logging.debug(f"[multimodal_agent_strategy] last message: {last_msg}")

            agent_selection = {
                "user": "triage_agent",
                "triage_agent": "multimodal_creator" if isinstance(last_msg, ToolCallSummaryMessage) else None,
                "multimodal_creator": "main_assistant",
                "main_assistant": "chat_closure",
            }

            selected_agent = agent_selection.get(last_msg.source)

            if selected_agent is None and last_msg.source == "triage_agent":
                selected_agent = "chat_closure"

            if selected_agent:
                logging.debug(f"[multimodal_agent_strategy] selected {selected_agent} agent")
                return selected_agent

            logging.debug("[multimodal_agent_strategy] selected None")
            return None
        
        self.selector_func = custom_selector_func

        self.agents = [triage_agent, multimodal_creator, main_assistant, chat_closure]
        
        return self._get_agents_configuration()
