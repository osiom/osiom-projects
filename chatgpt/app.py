import openai
import os

# Replace YOUR_API_KEY with your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

"""
# To see available models
data = openai.Engine.list()
print(data)
"""

# Set the model and prompt
engine = "text-davinci-003"
prompt = "Write a blog about ChatGPT"

# Set the maximum number of tokens to generate in the response
# Tokens can be thought of as pieces of words
max_tokens = 250

# Generate a response
completion = openai.Completion.create(
    engine=engine,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=0.5,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Print the response
print(completion.choices[0].text)