import psycopg2
import pgvector

from .vector_store import VectorStore
from configuration import Configuration

class PostgresVectorStore(VectorStore):
    def __init__(self, config:Configuration = None):
        super().__init__()
        # Initialize the PostgreSQL vector store

        self.config = config or Configuration()

        self.server = self.config.get("postgres", "server")
        self.database = self.config.get("postgres", "database")
        self.user = self.config.get("postgres", "user")
        self.password = self.config.get("postgres", "password")
        self.port = self.config.get("postgres", "port", default=5432)
        
        self.connection_string = f"postgresql://{self.user}:{self.password}@{self.server}:{self.port}/{self.database}"

    def connect(self):
        # Connect to the PostgreSQL database
        try:
            self.connection = psycopg2.connect(self.connection_string)
            self.cursor = self.connection.cursor()
            print("Connected to PostgreSQL database.")
        except Exception as e:
            print(f"Error connecting to PostgreSQL database: {e}")
        
    def disconnect(self):
        # Disconnect from the PostgreSQL database
        if hasattr(self, 'connection'):
            self.cursor.close()
            self.connection.close()
            print("Disconnected from PostgreSQL database.")

    async def add_vector(self, vector: list):
        # Add a vector to the PostgreSQL store
        pass

    async def query_vector(self, query: list):
        # Query the PostgreSQL vector store
        pass
