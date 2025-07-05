from huggingface_hub import InferenceClient
import os

def is_valid_query(query: str) -> bool:
    client = InferenceClient(provider="together", api_key=os.getenv("HUGGINGFACE_API_KEY"))

    system_prompt = (
        "You are a query classifier and you return return only VALID or INVALID after analyse the query.\n"
        "Your task is to decide whether a user input is a valid web search query or not.\n\n"
        "A valid query asks for factual information, explanations, or general knowledge (e.g., 'What is quantum computing?').\n"
        "An invalid query is a command, order, task, or to-do (e.g., 'walk my pet, add apples to grocery', 'add milk to grocery list', 'walk the dog', 'remind me').\n\n"
        "If the query is invalid return INVALID and if the query is valid return VALID"
    )

    prompt = f"""
    Use the following system prompt to generate a detailed and unified summary:

    SYSTEM PROMPT: {system_prompt}

    COMBINED CONTENT:
    {query[:100]}
    """

    stream = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=True,
    )

    output = ""
    for chunk in stream:
        output += chunk.choices[0].delta.content or ""

    result = output.strip().upper()
    return "VALID" in result
