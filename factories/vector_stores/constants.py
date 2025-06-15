from enum import Enum

class VectorStoreType(Enum):
    COSMOS = 'cosmos'
    ELASTICSEARCH = 'elasticsearch'
    AISEARCH = 'aisearch'
    POSTGRES = 'postgres'
    AZURE_SQL = 'azure_sql'