# ChatGPT Plugin for Brazilian Laws and Bills

This repository contains a plugin for ChatGPT that can retrieve relevant information about laws and bills in Brazil.

## Installation

To install and use this plugin with OpenAI's ChatGPT, follow these steps:

### 1. Set Up on Replit

- Create a new repl on Replit.com.
- Clone this repository into your repl.
- pip install -r requirements.txt
- Modify the URLs in the code to point to your repl's URL.

### 2. Configure OpenAI API Key

In your repl, set up the OpenAI API env key in Secrets section:

'''python
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
'''
### 3. Integrate with ChatGPT

- Go to ChatGPT (GPT-4) and navigate to Plugins.
- Select "Develop your own Plugin".
- Paste the URL related to your repl project.
- When prompted for the _SERVICE_AUTH_KEY in main.py file that is= 'hello'.

### 4. Verification

Copy the verification code provided by ChatGPT. Paste this code into `.well-known/ai-plugin.json` in the `verification_tokens` section:

'''json
{
  "verification_tokens": {
    "openai": "OPENAI-PLUGIN-CODE-HERE"
  }
}'''

###  5. Enable the Plugin
Once everything is set up, enable the plugin in ChatGPT to start using it.

## Usage
After installation, you can use the plugin within ChatGPT to retrieve information about Brazilian laws and bills by invoking the plugin's specific commands or queries.

## Official Website
For more information about this plugin and other related projects, [My WebSite](https://dametodata.com)

## Plugin Information
Detailed information about the TalkLawBrazil plugin can be found on our official page:[Detailed Plugin Information](https://www.dametodata.com/plugins-for-gpt)

## Chat Demo
See the capabilities of the TalkLawBrazil plugin: [Chat Demo](https://chat.openai.com/share/72b92285-1bda-4419-9657-61a28ca9982a)

