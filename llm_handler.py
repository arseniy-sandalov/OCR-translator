import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.prompts import PromptTemplate

class LLMHandler:
    def __init__(self, repo_id: str, filename: str, max_tokens: int, temperature: float):

        self.model_path = self.download_model(repo_id, filename)
        self.model = Llama(model_path=self.model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.template = """
            You are an expert in spelling correction and OCR post-processing. Your task is to correct the spelling of the word according to the product description. Here is the project description: {product_description} and the word: {ocr_word} Output ONLY the corrected word.
            """

    def download_model(self, repo_id: str, filename: str) -> str:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return model_path

    def generate_response(self, product_description: str, ocr_word: str) -> str:
        # Format the prompt
        prompt_template = PromptTemplate.from_template(self.template)
        prompt = prompt_template.format(product_description=product_description, ocr_word=ocr_word)

        # Generate response from LLM
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response
