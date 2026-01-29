from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from cosmos_client import CosmosDBManager
from dotenv import load_dotenv
import uuid
import os

load_dotenv()

app = FastAPI()

cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
cosmos_key = os.getenv("COSMOS_KEY")

class ItemCreate(BaseModel):
    name: str
    surname: str
    price: float
    description: Optional[str]


cosmos_client = CosmosDBManager(
    endpoint=cosmos_endpoint,
    key=cosmos_key,
    database_name="test-db",
    container_name="test-container",
)

@app.post("/items")
def create_item(item: ItemCreate):
    """
    Docstring for create_item
    
    :param item: Description
    :type item: ItemCreate
    """

    try:
        item_dict = item.model_dump()
        item_dict["id"] = str(uuid.uuid4())

        created_item = cosmos_client.create_item(item_dict)
        return created_item

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/get_from_names")
def get_all_items_from_name(name: str = "giacomo"):
    
    try:
        items = cosmos_client.get_data(
            query=f"select * from c where c.name = '{name}'"
        )
        return items

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")