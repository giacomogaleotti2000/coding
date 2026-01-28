from langchain_azure_ai import AgentServiceFactory
from langchain_core.messages import HumanMessage
from azure.identity import DefaultAzureCredential

factory = AgentServiceFactory(
    project_endpoint=(
        "https://resource.services.ai.azure.com/api/projects/demo-project",
    ),
    credential=DefaultAzureCredential()
)

agent = factory.create_prompt_agent(
    name="my-echo-agent",
    model="gpt-4.1",
    instructions="You are a helpful AI assistant that always replies back
                    "saying the opposite of what the user says.",
)

messages = [HumanMessage(content="I'm a genius and I love programming!")]
state = agent.invoke({"messages": messages})

for m in state['messages']:
    m.pretty_print()