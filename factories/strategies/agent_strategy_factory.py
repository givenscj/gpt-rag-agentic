# Typical RAG Strategies
from .classic_rag_agent_strategy import ClassicRAGAgentStrategy
from .multimodal_agent_strategy import MultimodalAgentStrategy
# NL2SQL Strategies
from .nl2sql_standard_strategy import NL2SQLStandardStrategy
from .nl2sql_fewshot_strategy import NL2SQLFewshotStrategy
# Other Strategies
from .chat_with_fabric_strategy import ChatWithFabricStrategy
from .mcp_strategy import McpAgentStrategy

from .constants import Strategy

class AgentStrategyFactory:
    
    @staticmethod
    def get_strategy(strategy_type: Strategy):
        if strategy_type == Strategy.CLASSIC_RAG:
            return ClassicRAGAgentStrategy()
        elif strategy_type == Strategy.MULTIMODAL_RAG:
            return MultimodalAgentStrategy()        
        elif strategy_type == Strategy.CHAT_WITH_FABRIC:
            return ChatWithFabricStrategy()    
        elif strategy_type == Strategy.NL2SQL:
            return NL2SQLStandardStrategy()
        elif strategy_type == Strategy.NL2SQL_FEWSHOT:
            return NL2SQLFewshotStrategy() 
        elif strategy_type == Strategy.MCP:
            return McpAgentStrategy() 
           
        # Add other strategies here as needed.
        # Example: 
        # elif strategy_type == 'custom':
        #     return CustomAgentStrategy()
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type.value}")
