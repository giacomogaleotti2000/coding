import os
import json
from azure.cosmos import CosmosClient, exceptions
from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")
DATABASE_NAME = os.getenv("AZURE_COSMOS_DATABASE", "AgentDatabase")
CONTAINER_NAME = os.getenv("AZURE_COSMOS_CONTAINER", "Items")

def query_cosmos_db(query_text: str):
    """
    Executes a SQL query against the Azure Cosmos DB container.
    
    SCHEMA CONTEXT:
    The container contains the following fields, I paste the schema here as a reference with inside the objects the fields descriptions:
    {
    "id": " ", // unique identifier for the customer
    "Id": " ", // unique identifier for the customer
    "Conto": " ", // unique identifier for the customer account
    "Sottoconto": " ", // unique identifier for the customer account sub-account
    "AccountType_IsCustomer":  , // boolean flag to indicate if the account is a customer
    "Nome_Cliente": " ", // name of the customer
    "Indirizzo_Cliente": " ", // address of the customer
    "Codice_Postale_Cliente": " ", // postal code of the customer
    "Contact_IsoCode": " ", // ISO code of the customer's country
    "Citta_Cliente": " ", // city of the customer
    "Contact_Country_Id":  , // ID of the customer's country
    "Paese_Cliente": " ", // country of the customer
    "FullAddress_Cliente": " ", // full address of the customer
    "Company_Id":  , // ID of the company the customer belongs to
    "Nome_Fornitore": " ", // name of the supplier
    "updated_at": " ", // timestamp of the last update
    "status_c": " ", // status of the customer
    "Data_Ultimo_Ordine": " ", // date of the last order
    "Cliente_con_Ordini":  , // boolean flag to indicate if the customer has orders
    "Destinazione_Cliente_Query": " ", // destination of the customer for the query
    "latitude":  , // latitude of the customer's location
    "longitude":  , // longitude of the customer's location
    "google_maps_place_name": " ", // name of the customer's location on Google Maps
    "google_maps_link": " ", // link to the customer's location on Google Maps
    "status_code": " ", // status code of the customer
    "up_to_date":  , // boolean flag to indicate if the customer's data is up to date
    "geocoding_updated_at": " ", // timestamp of the last geocoding update
    }

    Args:
        query_text: str - the SQL query to execute
    Returns:
        str - a text saying how many items were retrieved
    """
    if not COSMOS_ENDPOINT or not COSMOS_KEY:
        return {"error": "Cosmos DB credentials not configured in environment variables."}

    try:
        client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        database = client.get_database_client(DATABASE_NAME)
        container = database.get_container_client(CONTAINER_NAME)

        print(f"\n[Debug] Querying DB: {DATABASE_NAME}, Container: {CONTAINER_NAME}")
        print(f"[Debug] SQL: {query_text}")

        # Execute the query
        items = list(container.query_items(
            query=query_text,
            enable_cross_partition_query=True
        ))

        # Save results to a JSON file as requested
        with open("cosmos_results.json", "w") as f:
            json.dump(items, f, indent=4)
        
        print(f"query results: \n{json.dumps(items, indent=4)}")
        
        return f"Successfully retrieved {len(items)} items."

    except exceptions.CosmosHttpResponseError as e:
        return {"error": f"An error occurred: {e.message}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
