from llm_handler import LLMHandler

REPO_ID = "QuantFactory/Dolphin3.0-Llama3.2-1B-GGUF"
FILENAME = "Dolphin3.0-Llama3.2-1B.Q4_K_M.gguf"
def main():
    llm = LLMHandler(repo_id=REPO_ID, filename=FILENAME)

    # Call LLM and get response
    response = llm.generate_response(country="France")

    print("Model Response:", response)

if __name__ == "__main__":
    main()
