# Import the required agents
from crewai import Agent
from new_agents import ResearchAgent, TheoryTestingAgent, SuggestionAgent

# Assuming the core_gpt_model and all agent classes (ResearchAgent, TheoryTestingAgent, SuggestionAgent) are defined above

def genai_research_assistant(agent_type: str, input_data: str) -> str:
    """
    This function integrates all the agents (Research, Theory Testing, Suggestion) and directs tasks to the appropriate agent.
    
    :param agent_type: The type of agent to be used ('research', 'theory_testing', 'suggestion').
    :param input_data: The query, theory, or research topic that the agent will process.
    :return: The output from the chosen agent based on the input data.
    """
    # Initialize all agents
    agents = {
        "research": ResearchAgent(),
        "theory_testing": TheoryTestingAgent(),
        "suggestion": SuggestionAgent()
    }
    
    # Check if the requested agent exists
    if agent_type in agents:
        # Delegate task to the correct agent and return the result
        return agents[agent_type].perform_task(input_data)
    else:
        return "Error: Invalid agent type. Please choose from 'research', 'theory_testing', or 'suggestion'."
    
# Example usage of the integrated GenAI research assistant

# Research Task
research_topic = "marine biodiversity"
result_research = genai_research_assistant(agent_type="research", input_data=research_topic)
print(f"Research Results for '{research_topic}':\n{result_research}")

# Theory Testing Task
theory = "Increased sea temperatures lead to higher fish mortality rates."
result_theory = genai_research_assistant(agent_type="theory_testing", input_data=theory)
print(f"Theory Testing Results for '{theory}':\n{result_theory}")

# Suggestion Task
research_topic = "sustainable fishing methods"
result_suggestion = genai_research_assistant(agent_type="suggestion", input_data=research_topic)
print(f"Suggestions for '{research_topic}':\n{result_suggestion}")

