# Multi-Model Chat GUI with PDF Context & Temperature Control

This is a Python-based GUI application that allows users to interact with various large language models (LLMs) like **GPT-4**, **LLAMA 3**, and **Groq**. The app supports PDF document parsing for contextual input, temperature control for response variability, and role-based system prompts. It is built using `tkinter` for a simple desktop interface.

## Features

- Supports multiple LLMs (GPT-4, LLAMA 3, Groq)
- Upload PDF files for contextual reference
- Role-based system prompts
- Adjustable temperature for response creativity
- Streamed or generated text output display
- Built-in GUI using `tkinter`

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llm-multimodel-chat-gui.git
cd llm-multimodel-chat-gui
```

### 2. Install Dependencies

```bash
pip install -U "transformers==4.40.0"
pip install -i https://pypi.org/simple/ bitsandbytes
pip install accelerate
pip install openai
pip install PyPDF2
pip install groq
```

*Note: Make sure you have `torch` installed, and a compatible CUDA version if using GPU.*

## LLAMA 3 Reference Setup

This project uses the LLAMA 3 model via `transformers` and `unsloth`. For setup and code reference, check this Colab:
[LLAMA3 Code Reference](https://colab.research.google.com/drive/1Z2x2ujqyGeefSUPGp31ij5dnrOSvoifq?usp=sharing)

## API Keys

To use **OpenAI GPT-4** or **Groq** APIs, replace the placeholders in the code with your actual API keys:

```python
client = openai.OpenAI(api_key="your_openai_key")
client = Groq(api_key="your_groq_key")
```

## How to Use

1. Run the script:
    ```bash
    python app.py
    ```

2. In the GUI:
    - Choose your model (GPT-4, LLAMA 3, or Groq)
    - Enter a system role (e.g., “You are a helpful assistant”)
    - Input your message
    - Optionally attach PDF files
    - Adjust temperature (0.0 to 2.0)
    - Click **Submit** to interact

3. Use **Refresh** to clear the conversation or **Select Files** to upload more PDFs.

## Screenshots

*(Add screenshots here if needed)*

## License

MIT License. See [LICENSE](LICENSE) for more info.
