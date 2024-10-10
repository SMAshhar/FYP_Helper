import os
from crewai import Agent
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)

# Function to interact with GPT-o1
def core_gpt_model(query: str) -> str:
    """
    This function interacts with the GPT-o1 model and returns the response.
    
    :param query: The query or task for GPT-o1 to process.
    :return: Processed response from the model.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                    ],
                }
            ],
        )
                # Return the model's response
        return response['choices'][0]['text'].strip()
    
    except Exception as e:
        return f"Error interacting with GPT-o1 model: {str(e)}"


class OldFypAgent(BaseModel):
    role: str = "Previous Final Year Projects Assistant"
    goal: str = "Fetch relevant papers, projects and information based on the query."

    def perform_task(self, fyp_topic: str) -> str:
        query = f"Find papers, projects and studies on {fyp_topic}. Provide relevant findings and insights."
        return core_gpt_model(query)

class ProjectAdvisorAgent(BaseModel):
    role: str = "Project Advisor"
    goal: str = "Suggest improvements or new directions based on existing work."

    def perform_task(self, research_topic: str) -> str:
        query = f"Provide suggestions and improvements for research on {research_topic}. Recommend new directions or areas of focus."
        return core_gpt_model(query)
