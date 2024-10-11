# Functions for the research agent
def research_fyps(call:str):
    # describe the function to call the data base
    result = "<desctibe the function details here>"
    return result


# Functions for the theory testing agent
def advisory_call(call:str):
    # describe the function to call the data base
    result = "<desctibe the function details here>"
    return result


tools_func = [
    {
        "type": "function",
        "function": {
            "name": "research_fyps",
            "description": "Retrieve relevant research papers and studies related to the given research topic. Call this function whenever a researcher requests information on previously published work, including papers, studies, or related resources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "call": {
                        "type": "string",
                        "description": "The research topic or field for which relevant papers and studies are needed.",
                    }
                },
                "required": ["call"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advisory_call",
            "description": "Analyze and test a proposed theory to validate its soundness. Call this function when a researcher needs to test or verify a specific theory and receive conclusions or insights from previous research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "call": {
                        "type": "string",
                        "description": "The theory that needs to be analyzed or tested for its validity and soundness.",
                    }
                },
                "required": ["call"],
            },
        },
    },
]