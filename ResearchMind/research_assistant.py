# Import the required agents
from new_agents import OldFypAgent, ProjectAdvisorAgent


def genai_research_assistant(agent_type: str, input_data: str) -> str:
    """
    This function integrates all the agents (history_of_fyp, advisor) and directs tasks to the appropriate agent.
    
    :param agent_type: The type of agent to be used ('OldFypAgent','ProjectAdvisorAgent').
    :param input_data: The query, theory, or research topic that the agent will process.
    :return: The output from the chosen agent based on the input data.
    """
    # Initialize all agents
    agents = {
        "OldFypAgent": OldFypAgent(),
        "ProjectAdvisorAgent": ProjectAdvisorAgent()
    }
    
    # Check if the requested agent exists
    if agent_type in agents:
        # Delegate task to the correct agent and return the result
        return agents[agent_type].perform_task(input_data)
    else:
        return "Error: Invalid agent type. Please choose from 'research', 'theory_testing', or 'suggestion'."
    
# Example usage of the integrated GenAI research assistant

# Previous Final Year Projects Task
fyp_topic = "Study the impact of acid rain on concrete infused with PVC"
result_research = genai_research_assistant(agent_type="research", input_data=fyp_topic)
print(f"Research Results for '{fyp_topic}':\n{result_research}")


# Project Advisor Task
fyp_topic = "Study the impact of acid rain on concrete infused with PVC"
result_suggestion = genai_research_assistant(agent_type="suggestion", input_data=fyp_topic)
print(f"Suggestions for '{fyp_topic}':\n{result_suggestion}")

