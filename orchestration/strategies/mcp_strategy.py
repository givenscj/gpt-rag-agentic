from typing import Annotated


from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, SseServerParams

from tools import get_time, get_today_date
from tools import vector_index_retrieve
from tools.ragindex.types import VectorIndexRetrievalResult

from .base_agent_strategy import BaseAgentStrategy
from ..constants import Strategy

from autogen_agentchat.messages import ToolCallSummaryMessage

from semantic_kernel.connectors.mcp import MCPSsePlugin

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from configuration import Configuration

class McpAgentStrategy(BaseAgentStrategy):

    def __init__(self):
        super().__init__()
        self.strategy_type = Strategy.MCP
        self.config = Configuration()

    async def create_agents(self, history, client_principal=None, access_token=None, output_mode=None, output_format=None):
        """
        MCP creation strategy that creates the basic agents and registers MCP server+functions.
        
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
        mcp_server_url = self.config.get_value("AZURE_MCP_SERVER_URL", default="http://localhost:5000")
        mcp_server_timeout = self.config.get_value("AZURE_MCP_SERVER_TIMEOUT", default=30)
        mcp_server_api_key = self.config.get_value("AZURE_MCP_SERVER_APIKEY", default=None)

        headers={"Content-Type": "application/json"},

        if mcp_server_api_key is not None:
            headers['Authorization'] = f"Bearer {mcp_server_api_key}"

        server_params = SseServerParams(
            url=mcp_server_url,
            headers=headers,
            timeout=mcp_server_timeout,  # Connection timeout in seconds
        )

        # Agents

        ## Main Assistant Agent
        '''
        server_params = StdioServerParams(
            command="npx",
            args=[
                "@playwright/mcp@latest",
                "--headless",
            ],
        )
        '''

        #https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.tools.mcp.html
        async with McpWorkbench(server_params) as mcp:
            assistant_prompt = await self._read_prompt("main_assistant")
            main_assistant = AssistantAgent(
                name="main_assistant",
                system_message=assistant_prompt,
                model_client=self._get_model_client(), 
                workbench=mcp,
                #tools=[get_today_date, get_time], #tools can't be used with a workbench
                reflect_on_tool_use=True,
                model_context=shared_context
            )

            ## Chat Closure Agent
            chat_closure = await self._create_chat_closure_agent(output_format, output_mode)

        # Agent Configuration

        # Optional: Override the termination condition for the assistant. Set None to disable each termination condition.
        # self.max_rounds = int(os.getenv('MAX_ROUNDS', 8))
        # self.terminate_message = "TERMINATE"

        # Optional: Define a selector function to determine which agent to use based on the user's ask.
        def custom_selector_func(messages):
            """
            Selects the next agent based on the source of the last message.
            
            Transition Rules:
               user -> assistant
               assistant -> None (SelectorGroupChat will handle transition)
            """
            last_msg = messages[-1]
            if last_msg.source == "user":
                return "main_assistant"
            if last_msg.source == "main_assistant" and isinstance(last_msg, ToolCallSummaryMessage):
                return "main_assistant"
            if last_msg.source in ["main_assistant"]:
                return "chat_closure"                 
            return None

        
        self.selector_func = custom_selector_func
        
        self.agents = [main_assistant, chat_closure]
        
        return self._get_agents_configuration()
