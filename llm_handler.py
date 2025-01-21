import os
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

class LLMHandler:
    def __init__(self, repo_id: str, max_tokens: int, temperature: float, hf_api_token: str):
        self.llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"max_new_tokens": max_tokens, "temperature": temperature},
            huggingfacehub_api_token=hf_api_token
        )
        self.template = """
            Вы эксперт по исправлению орфографии и постобработке OCR. Ваша задача — исправить орфографию слова. Вот слово: {ocr_word}. Выведите ТОЛЬКО исправленное слово!.
            """

    def generate_response(self, product_description: str, ocr_word: str) -> str:
        # Format the prompt
        prompt_template = PromptTemplate.from_template(self.template)
        prompt = prompt_template.format(product_description=product_description, ocr_word=ocr_word)

        # Generate response from LLM
        response = self.llm(prompt)

        return response.strip()

