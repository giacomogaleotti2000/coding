from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv

# load env variables from .env file
load_dotenv()

def main():
    
    print("Hello from agents!")

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=LitellmModel(model="azure_ai/DeepSeek-V3.2")
        )

    result = Runner.run_sync(agent, "what's the weather like today?")
    print(result.final_output)


if __name__ == "__main__":
    main()