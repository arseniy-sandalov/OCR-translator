from llm_handler import LLMHandler

HF_MODEL_PATH = "/QuantFactory/Dolphin3.0-Llama3.2-1B-GGUF/blob/main/Dolphin3.0-Llama3.2-1B.Q4_K_M.gguf"
def main():
    llm = LLMHandler(hf_model_path=HF_MODEL_PATH)

    # Call LLM and get response
    response = llm.generate_response(country="France")

    print("Model Response:", response)

if __name__ == "__main__":
    main()
