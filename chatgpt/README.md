# ChatGPT

Instructions to connect and run python code integrated with ChatGPT.

By default at a new user creation we get assigned 18$ in credit, those are what we need in order to make the API calls.

## Create account

Go to [this](https://platform.openai.com/account/api-keys) website and register to find your API keys.

Set the copied Keys from the previous step as env variable
```
export OPENAI_API_KEY=<your-openai-key>
```

## Run the code
```
python app.py
```

# Notes

The code designed in this repository is meant to be used to connect with the old models and engine developed by OpenAI.

To connect and run interactive code with ChatGPT3 the only available way and python code is to follow [this](https://github.com/mmabrouk/chatgpt-wrapper) guidelines.