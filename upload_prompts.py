import os
import json

from connectors import CosmosDBClient
from configuration import Configuration

config = Configuration()

client = CosmosDBClient(config=config)

#read all directories from the prompts directory
prompts_directory = os.path.join(os.getcwd(), "prompts")

for root, dirs, files in os.walk(prompts_directory):
    for file in files:
        file_path = os.path.join(root, file)
        
        try:
            with open(f"{file_path}", 'r', encoding='utf-8') as f:
                content = f.read()
                # Assuming the content is a JSON string
                #get the parent directory name to use as the ID prefix
                dir_name = os.path.basename(root)
                data = {
                    "id" : f"{dir_name}_{os.path.splitext(file)[0]}",  # Use the file name without extension as the ID
                    "content": content  # Parse the JSON content
                }
                client.create_document("prompts", data["id"], body=data)
                #client.update_document("prompts", data)
        except Exception as e:
            print(f"Error {file_path}: {e}")