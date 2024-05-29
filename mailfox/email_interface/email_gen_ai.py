import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

class EmailLLM():
    # gpt-4-turbo-preview
    def __init__(self, api_key, model_name="gpt-3.5-turbo-0125"):
        self.model_name = model_name
        self.openai = OpenAI(api_key=api_key)
    
    def generate_labels(self, emails):
        # Format the emails so they're listed individually so the LLM can understand them better
        formatted_emails = [f"Email {i}: {email['subject']}\n {email['body']}" for i, email in enumerate(emails)]

        # Create a chat with GPT-3.5
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": f"You are an email assistant that classifies emails into catagories. Given a list of emails respond with a catagory that characterizes the emails the best and nothing else."},
                {"role": "user", "content": formatted_emails}
            ]
        )

        # Get the category from the response
        category = response.choices[0].message.content
        
        return category
    
    def predict_folder(self, email, folders):
        formatted_email = f"{email['from']} -> {email['to']}\n{email['subject']}\n{email['body']}"
        
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": f"You are an email assistant that classifies emails into one of these categories: {folders}. Only respond with the email catagory and nothing else."},
                {"role": "user", "content": formatted_email}
            ]
        )
        
        folder = response.choices[0].message.content
        
        if folder.strip()[0] != '"':
            folder = '"' + folder + '"'
        
        if folder.strip() not in folders:
            raise ValueError(f"Folder {folder} not in {folders}")
        
        return folder