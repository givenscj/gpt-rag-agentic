from .constants import VectorStoreType
from .cosmos import CosmosVectorStore
from .elasticsearch import ElasticSearchVectorStore
from .aisearch import AISearchVectorStore
from .postgres import PostgresVectorStore
from .azure_sql import AzureSqlVectorStore

class VectorStoreFactory:
    
    @staticmethod
    def get_factory(strategy_type: VectorStoreType):
        if strategy_type == VectorStoreType.COSMOS:
            return CosmosVectorStore()
        elif strategy_type == VectorStoreType.ELASTICSEARCH:
            return ElasticSearchVectorStore()
        elif strategy_type == VectorStoreType.AISEARCH:
            return AISearchVectorStore()
        elif strategy_type == VectorStoreType.POSTGRES:
            return PostgresVectorStore()
        elif strategy_type == VectorStoreType.AZURE_SQL:
            return AzureSqlVectorStore()
        
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type.value}")