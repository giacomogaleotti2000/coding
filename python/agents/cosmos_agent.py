import os
import json
from agents import Agent, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from cosmos_db_tool import query_cosmos_db

# Load environment variables
load_dotenv()

def main():
    # Initialize the model
    # Using the same model configuration as in main.py
    model_name = os.getenv("AZURE_MODEL_NAME", "azure/gpt-5-nano")
    model = LitellmModel(model=model_name)

    # Wrap the function in function_tool to provide metadata required by the Agent
    query_tool = function_tool(query_cosmos_db)

    # Define the Cosmos DB Agent
    cosmos_agent = Agent(
        name="CosmosDataAgent",
        instructions=(
            "You are a data retrieval assistant. Your primary task is to help users query data from Azure Cosmos DB. "
            "When a user asks for data that requires a database lookup, use the 'query_cosmos_db' tool. "
            "You should generate valid Cosmos DB SQL queries based on the user's request. "
            "Assume the container name is 'Items' unless specified otherwise. "
            "After calling the tool, inform the user that the data has been retrieved and saved to 'cosmos_results.json'. "
            "Do NOT try to display the full data in the chat if it's large; just confirm retrieval. "
            "If the user asks a general question not related to data retrieval, answer it normally."
        ),
        model=model,
        tools=[query_tool]
    )

    print("--- Cosmos DB Agent Started ---")
    print("Type 'exit' or 'quit' to stop.")

    # Simple in-memory chat history (json-like array)
    chat_history = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Add user message to history
        chat_history.append({"role": "user", "content": user_input})

        try:
            # Run the agent with the current input
            # The Runner handles tool calls automatically
            result = Runner.run_sync(cosmos_agent, user_input)
            
            # Print agent's response
            print(f"\nAgent: {result.final_output}")

            # Add agent response to history
            chat_history.append({"role": "assistant", "content": result.final_output})

            # Save chat history to a local JSON file for "memory"
            with open("chat_memory.json", "w") as f:
                json.dump(chat_history, f, indent=4)

        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
  