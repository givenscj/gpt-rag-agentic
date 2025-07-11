import os
from pydantic import BaseModel
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from .nl2sql_base_agent_strategy import NL2SQLBaseStrategy
from .constants import Strategy
from tools import (
    get_time,
    get_today_date,
    queries_retrieval,
    get_all_datasources_info,
    get_all_tables_info,
    get_schema_info,
    validate_sql_query,
    execute_sql_query,
)

from configuration import Configuration
from dependencies import get_config
config :Configuration = get_config()

# Agents Strategy Class

class NL2SQLFewshotStrategy(NL2SQLBaseStrategy):

    def __init__(self):
        self.strategy_type = Strategy.NL2SQL_FEWSHOT
        super().__init__()

        
    async def create_agents(self, history, client_principal=None, access_token=None, output_mode=None, output_format=None):
        """
        Creates agents and registers functions for the NL2SQL single agent scenario.
        """

        # Model Context
        shared_context = await self._get_model_context(history) 
        
        # Wrapper Functions for Tools

        get_all_datasources_info_tool = FunctionTool(
            get_all_datasources_info, description="Retrieve a list of all datasources."
        )

        get_all_tables_info_tool = FunctionTool(
            get_all_tables_info, description="Retrieve a list of tables filtering by the given datasource."
        )

        get_schema_info_tool = FunctionTool(
            get_schema_info, description="Retrieve information about tables and columns from the data dictionary."
        )        

        queries_retrieval_tool = FunctionTool(
            queries_retrieval, description="Retrieve QueriesRetrievalResult a list of similar QueryItem containing a question, the correspondent query and reasoning."
        )

        validate_sql_query_tool = FunctionTool(
            validate_sql_query, description="Validate the syntax of an SQL query."
        )     

        execute_sql_query_tool = FunctionTool(
            execute_sql_query, description="Execute an SQL query and return the results."
        )

        # Agents

        ## Assistant Agent
        assistant_prompt = await self._read_prompt("nl2sql_assistant")
        assistant = AssistantAgent(
            name="assistant",
            system_message=assistant_prompt,
            model_client=self._get_model_client(), 
            tools=[get_all_datasources_info_tool, get_schema_info_tool, validate_sql_query_tool, queries_retrieval_tool, get_all_tables_info_tool, execute_sql_query_tool, get_today_date, get_time],
            reflect_on_tool_use=True,
            model_context=shared_context
        )

        ## Chat Closure Agent
        chat_closure = await self._create_chat_closure_agent(output_format, output_mode)
        
        # Group Chat Configuration

        self.max_rounds = int(config.get_value('MAX_ROUNDS', 20))

        def custom_selector_func(messages):
            """
            Selects the next agent based on the source of the last message.
            
            Transition Rules:
               user -> assistant
               assistant -> None (SelectorGroupChat will handle transition)
            """
            last_msg = messages[-1]
            if last_msg.source == "user":
                return "assistant"
            else:
                return None     
        
        self.selector_func = custom_selector_func

        self.agents = [assistant, chat_closure]
        
        return self._get_agents_configuration()
