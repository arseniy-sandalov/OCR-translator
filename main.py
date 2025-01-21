from llm_handler import LLMHandler

MODEL_PATH = "path/to/your/model.gguf"

def main():
    llm = LLMHandler(model_path=MODEL_PATH)

    # Call LLM and get response
    response = llm.generate_response(country="France")

    print("Model Response:", response)

if __name__ == "__main__":
    main()
