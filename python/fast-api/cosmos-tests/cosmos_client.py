# cosmos_client.py
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from typing import Optional, Dict, Any
import os

class CosmosDBManager:
    def __init__(self, endpoint: str, key: str, database_name: str, container_name: str):
        """
        Initialize the Cosmos DB client
        
        Args:
            endpoint: Your Cosmos DB endpoint (e.g., https://yourdb.documents.azure.com:443/)
            key: Your Cosmos DB primary or secondary key
            database_name: Name of your database
            container_name: Name of your container
        """
        self.client = CosmosClient(endpoint, key)
        self.database_name = database_name
        self.container_name = container_name
        
        # Get or create database
        self.database = self.client.get_database_client(database_name)
        
        # Get or create container
        self.container = self.database.get_container_client(container_name)
    
    def create_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new item in Cosmos DB
        
        Args:
            item: Dictionary with your data. Must include 'id' field
        
        Returns:
            Created item with metadata
        """
        try:
            created_item = self.container.create_item(body=item)
            return created_item
        except exceptions.CosmosResourceExistsError:
            raise ValueError(f"Item with id '{item.get('id')}' already exists")
        except exceptions.CosmosHttpResponseError as e:
            raise Exception(f"Failed to create item: {e.message}")
    
    def read_single_item(self, item_id: str, partition_key: str) -> Dict[str, Any]:
        """
        Read a single item by ID
        
        Args:
            item_id: The id of the item
            partition_key: The partition key value
        
        Returns:
            The item data
        """
        try:
            item = self.container.read_item(item=item_id, partition_key=partition_key)
            return item
        except exceptions.CosmosResourceNotFoundError:
            raise ValueError(f"Item with id '{item_id}' not found")
    
    def get_data(self, query: str, parameters: Optional[list] = None) -> list:
        """
        Query items using SQL-like syntax
        
        Args:
            query: SQL query string (e.g., "SELECT * FROM c WHERE c.category = @category")
            parameters: List of parameter dicts (e.g., [{"name": "@category", "value": "electronics"}])
        
        Returns:
            List of matching items
        """
        items = list(self.container.query_items(
            query=query,
            parameters=parameters or [],
            enable_cross_partition_query=True
        ))
        return items
    
    def update_item(self, item_id: str, partition_key: str, updated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing item
        
        Args:
            item_id: The id of the item
            partition_key: The partition key value
            updated_data: Dictionary with updated fields
        
        Returns:
            Updated item
        """
        # First read the item
        existing_item = self.read_single_item(item_id, partition_key)
        
        # Merge updates
        existing_item.update(updated_data)
        
        # Replace the item
        updated_item = self.container.replace_item(item=item_id, body=existing_item)
        return updated_item
    
    def delete_item(self, item_id: str, partition_key: str) -> None:
        """
        Delete an item
        
        Args:
            item_id: The id of the item
            partition_key: The partition key value
        """
        try:
            self.container.delete_item(item=item_id, partition_key=partition_key)
        except exceptions.CosmosResourceNotFoundError:
            raise ValueError(f"Item with id '{item_id}' not found")