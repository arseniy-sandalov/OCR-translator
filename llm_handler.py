import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.prompts import PromptTemplate

class LLMHandler:
    def __init__(self, repo_id: str, filename: str, max_tokens: int = 512, temperature: float = 0.7):

        self.model_path = self.download_model(repo_id, filename)
        self.model = Llama(model_path=self.model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.template = """
            You are an expert in text correction and OCR post-processing. You will receive an array of words extracted from an image using OCR. Your task is to correct any spelling mistakes and adjust the words based on the given product description. If two or more words should be combined to form a meaningful term, merge and correct them accordingly. Only output the corrected words, separated by commas, without any explanations or extra formatting.

            Product Description: {product_description}
            OCR Extracted Words: {ocr_words}

            Output the corrected words in a comma-separated format.
            """

    def download_model(self, repo_id: str, filename: str) -> str:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return model_path

    def generate_response(self, country: str) -> str:
        # Format the prompt
        prompt_template = PromptTemplate.from_template(self.template)
        prompt = prompt_template.format(country=country)

        # Generate response from LLM
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response["choices"][0]["text"].strip()
