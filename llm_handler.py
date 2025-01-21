from llama_cpp import Llama
from langchain.prompts import PromptTemplate

class LLMHandler:
    def __init__(self, model_path: str, max_tokens: int = 512, temperature: float = 0.7):
        """
        Initializes the Llama model.
        
        :param model_path: Path to the GGUF model file.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Sampling temperature for randomness in output.
        """
        self.model = Llama(model_path=model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.template = "What is the capital of {country}?"

    def generate_response(self, country: str) -> str:
        """
        Formats the prompt, runs inference, and returns the response.

        :param country: The country to ask about.
        :return: The model's response as a string.
        """
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
