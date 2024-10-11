import os
from crewai import Agent
from pydantic import BaseModel
from openai import OpenAI
from typing import List
from functions import tools_func

fyp_historian = [
    {
        "role": "system",
        "content": (
            "The Previous Final Year Projects Assistant is responsible for helping students and researchers locate relevant "
            "information from past final year projects (FYPs). This includes fetching related papers, projects, and case studies "
            "based on the research topic provided. The goal of the agent is to ensure that users can access relevant insights, "
            "findings, and methodologies from similar past projects to aid in their own research or project development. By "
            "leveraging existing data, the agent can streamline the research process and reduce the likelihood of redundant work, "
            "ensuring that no significant work from past projects is overlooked."
        )
    },
]
project_advisor = [
    {
        "role": "system",
        "content": (
            "The Project Advisor Agent serves as a virtual advisor, assisting researchers and students by offering suggestions "
            "for improving or expanding their current projects. The agent analyzes the research topic or project and, using the "
            "knowledge of existing work in the domain, suggests new directions, additional areas of focus, or potential improvements. "
            "Its goal is to inspire innovation and guide the researcher towards making impactful contributions, ensuring that the "
            "research or project is well-rounded, up-to-date, and positioned to address critical gaps in the field. This agent "
            "helps researchers think beyond their immediate focus and explore opportunities for growth."
        )
    }
]


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

def core_gpt_model(messages: List) -> str:
    """
    This function interacts with the GPT model and returns the response.
    
    :param messages: List of messages in chat format to be passed to the model.
    :return: Processed response from the model.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_func
        )
        # Return the model's response (strip any leading/trailing whitespace)
        return response['choices'][0]['message']['content'].strip()
    
    except Exception as e:
        return f"Error interacting with GPT model: {str(e)}"


class OldFypAgent(BaseModel):
    role: str = "Previous Final Year Projects Assistant"
    goal: str = (
        "The Previous Final Year Projects Assistant is responsible for helping students and researchers locate relevant "
        "information from past final year projects (FYPs). This includes fetching related papers, projects, and case studies "
        "based on the research topic provided. The goal of the agent is to ensure that users can access relevant insights, "
        "findings, and methodologies from similar past projects to aid in their own research or project development. By "
        "leveraging existing data, the agent can streamline the research process and reduce the likelihood of redundant work, "
        "ensuring that no significant work from past projects is overlooked."
    )

    def perform_task(self, fyp_project: str) -> str:
        # Create the user query
        user_query = {
            "role": "user",
            "content": f"Find research papers and studies on {fyp_project}. Provide relevant findings and insights."
        }
        
        # Append user query to the message list
        fyp_historian.append(user_query)
        
        # Get result from GPT model
        result = core_gpt_model(fyp_historian)
        
        # Append GPT's response as 'assistant' role in the conversation
        fyp_historian.append({"role": "assistant", "content": result})
        
        # Return the result
        return result

class ProjectAdvisorAgent(BaseModel):
    role: str = "Project Advisor"
    goal: str = (
        "The Project Advisor Agent serves as a virtual advisor, assisting researchers and students by offering suggestions "
        "for improving or expanding their current projects. The agent analyzes the research topic or project and, using the "
        "knowledge of existing work in the domain, suggests new directions, additional areas of focus, or potential improvements. "
        "Its goal is to inspire innovation and guide the researcher towards making impactful contributions, ensuring that the "
        "research or project is well-rounded, up-to-date, and positioned to address critical gaps in the field. This agent "
        "helps researchers think beyond their immediate focus and explore opportunities for growth."
    )

    def perform_task(self, fyp_project: str) -> str:
        # Create the user query
        user_query = {
            "role": "user",
            "content": f"Provide suggestions and improvements for research on {fyp_project}. Recommend new directions or areas of focus."
        }
        
        # Append user query to the message list
        fyp_historian.append(user_query)
        
        # Get result from GPT model
        result = core_gpt_model(fyp_historian)
        
        # Append GPT's response as 'assistant' role in the conversation
        fyp_historian.append({"role": "assistant", "content": result})
        
        # Return the result
        return result
