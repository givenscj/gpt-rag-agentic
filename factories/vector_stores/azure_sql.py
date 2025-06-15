from .vector_store import VectorStore

class AzureSqlVectorStore(VectorStore):
    def __init__(self):
        super().__init__()
        # Initialize the Azure SQL vector store
        pass

    def add_vector(self, vector: list):
        # Add a vector to the Azure SQL store
        pass

    def query_vector(self, query: list):
        # Query the Azure SQL vector store
        pass