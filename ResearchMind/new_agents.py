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


class ResearchAgent(BaseModel):
    role: str = "Research Assistant"
    goal: str = "Fetch relevant research papers and information based on the query."

    def perform_task(self, research_topic: str) -> str:
        query = f"Find research papers and studies on {research_topic}. Provide relevant findings and insights."
        return core_gpt_model(query)

class TheoryTestingAgent(BaseModel):
    role: str = "Theory Tester"
    goal: str = "Test the given theory and provide insights."

    def perform_task(self, theory: str) -> str:
        query = f"Analyze and test the following theory: {theory}. Provide insights and conclusions."
        return core_gpt_model(query)

class SuggestionAgent(BaseModel):
    role: str = "Research Advisor"
    goal: str = "Suggest improvements or new directions based on existing research."

    def perform_task(self, research_topic: str) -> str:
        query = f"Provide suggestions and improvements for research on {research_topic}. Recommend new directions or areas of focus."
        return core_gpt_model(query)
