import os
import openai
import langchain
import huggingface_hub
import transformers
import chromadb


def main():
    print("hello abc def")
    print(f"OpenAI version: {openai.version.__version__}")
    print(f"Langchain: {langchain.__version__}")
    print(f"HuggingFace hub: {huggingface_hub.__version__}")
    print(f"Transformer: {transformers.__version__}")
    print(f"ChromaDb: {chromadb.__version__}")


if __name__ == '__main__':
    main()
