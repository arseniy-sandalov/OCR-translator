from llm_handler import LLMHandler

HF_MODEL_PATH = "unsloth/Llama-3.2-1B-bnb-4bit"
def main():
    llm = LLMHandler(hf_model_path=HF_MODEL_PATH)

    # Call LLM and get response
    response = llm.generate_response(country="France")

    print("Model Response:", response)

if __name__ == "__main__":
    main()
