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
            You are an expert in spelling correction and OCR post-processing. You will receive a single word extracted from an image using OCR. Your task is to correct its spelling.

                If the word is misspelled, correct it.
                If you are unsure about the correct spelling, refer to the product description for context.
                If you still cannot determine the correct spelling, output the word as it is.

            Only output the corrected word without any explanations or extra formatting.

            Example:

                Product Description: "Chocolate-flavored protein bar"
                OCR Extracted Word: "protien"
                Output: "protein"

            Product Description: {product_description}
            OCR Extracted Word: {ocr_word}

            Output the corrected word.
            """

    def download_model(self, repo_id: str, filename: str) -> str:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return model_path

    def generate_response(self, product_description: str, ocr_words: str) -> str:
        # Format the prompt
        prompt_template = PromptTemplate.from_template(self.template)
        prompt = prompt_template.format(product_description=product_description, ocr_words=ocr_words)

        # Generate response from LLM
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response["choices"][0]["text"].strip()
