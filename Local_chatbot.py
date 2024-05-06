# LLAMA3 code Reference: https://colab.research.google.com/drive/1Z2x2ujqyGeefSUPGp31ij5dnrOSvoifq?usp=sharing
# pip install -U "transformers==4.40.0" --upgrade
# pip install -i https://pypi.org/simple/ bitsandbytes
# pip install accelerate


import tkinter as tk
from tkinter import scrolledtext, Scale, HORIZONTAL, filedialog
import openai
from PyPDF2 import PdfReader
import os

# Set your API key
client = openai.OpenAI(api_key="Your Key Here")
messages = []  # Global list to keep track of all conversation history
last_role = None  # To track last used role
last_file_paths = []  # To track last processed files


def process_pdf(file_path):
    file_name = os.path.basename(file_path)
    reader = PdfReader(file_path)
    document_pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_text = f"File name= {file_name}, Page= {page_number}: {text}"
        document_pages.append(page_text)
    return document_pages


def submit_action(event=None):
    global last_role, last_file_paths, file_paths
    role = role_entry.get()
    user_input = message_entry.get()
    current_temperature = temperature_scale.get()


    conversation_display.config(state=tk.NORMAL)
    conversation_display.insert(tk.END, f"\n\nYou: {user_input}")
    conversation_display.insert(tk.END, "\nBot: ")
    conversation_display.config(state=tk.DISABLED)
    message_entry.delete(0, tk.END)

    process_input(role, user_input, current_temperature)
    last_role = role
    last_file_paths = file_paths.copy()


def process_input(role, user_input, temperature):
    global last_role, last_file_paths, messages
    if role != last_role:
        messages.append({"role": "system", "content": role})
    if file_paths != last_file_paths:
        for path in file_paths:
            document_pages = process_pdf(path)
            for page_text in document_pages:
                messages.append({"role": "system", "content": page_text})
    messages.append({"role": "user", "content": user_input})

    selected_model = model_var.get()
    if selected_model == "GPT-4":
        response = client.chat.completions.create(
            model="gpt-4-1106-vision-preview",
            temperature=temperature,
            messages=messages,
            stream=True,
            max_tokens=4095
        )

        def update_conversation():
            try:
                content = next(response)
                if content.choices[0].delta.content is not None:
                    conversation_display.config(state=tk.NORMAL)
                    conversation_display.insert(tk.END, f"{content.choices[0].delta.content}")
                    conversation_display.config(state=tk.DISABLED)
                    root.after(10, update_conversation)
            except StopIteration:
                return
            except Exception as e:
                conversation_display.config(state=tk.NORMAL)
                conversation_display.insert(tk.END, f"\nError: {str(e)}\n")
                conversation_display.config(state=tk.DISABLED)

        update_conversation()

    elif selected_model == "LLAMA 3":
        import transformers
        import torch

        model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        response = pipeline(
            prompt,
            max_new_tokens=1000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
        conversation_display.config(state=tk.NORMAL)
        conversation_display.insert(tk.END, response[0]["generated_text"][len(prompt):])
        conversation_display.config(state=tk.DISABLED)


def reset_gui():
    global file_paths
    file_paths = []
    role_entry.delete(0, tk.END)
    message_entry.delete(0, tk.END)
    temperature_scale.set(1.0)
    file_label.config(text="No files selected")
    conversation_display.config(state=tk.NORMAL)
    conversation_display.delete(1.0, tk.END)
    conversation_display.config(state=tk.DISABLED)

def select_attachment():
    global file_paths
    # Convert the tuple to a list right when you assign it
    file_paths = list(filedialog.askopenfilenames(title="Select Files", filetypes=(("PDF files", "*.pdf"), ("All files", "*.*"))))
    num_files = len(file_paths)
    file_label.config(text=f"{num_files} files selected" if num_files else "No files selected")



def reset_and_refresh():
    reset_gui()  # Reset all GUI components first
    global last_role, last_file_paths, messages
    messages = []  # Clear the message history  # Then refresh messages if needed
    last_role = None
    last_file_paths = []

file_paths = []
# Define the main application window
root = tk.Tk()
root.title("Role and Message Input Interface with Temperature Control and File Selection")


# Main conversation display
conversation_display = scrolledtext.ScrolledText(root, height=15, width=80)
conversation_display.pack(pady=10, fill=tk.BOTH, expand=True)
conversation_display.config(state=tk.DISABLED)

# Role entry section
role_label = tk.Label(root, text="Enter Your Role:")
role_label.pack(fill=tk.X)
role_entry = tk.Entry(root)
role_entry.pack(fill=tk.X, padx=5)

# Message entry section
message_label = tk.Label(root, text="Enter Your Message:")
message_label.pack(fill=tk.X)
message_entry = tk.Entry(root)
message_entry.pack(fill=tk.X, padx=5, pady=5)
message_entry.bind("<Return>", submit_action)  # Bind the Enter key to submit action

# model selection
model_options = ["GPT-4", "LLAMA 3"]
model_var = tk.StringVar(root)
model_var.set(model_options[0])  # default value
model_label = tk.Label(root, text="Select Model:")
model_label.pack(fill=tk.X)
model_dropdown = tk.OptionMenu(root, model_var, *model_options)
model_dropdown.pack(fill=tk.X, padx=5)

# Temperature control
temperature_label = tk.Label(root, text="Select Temperature:")
temperature_label.pack(fill=tk.X)
temperature_scale = Scale(root, from_=0.0, to=2.0, resolution=0.01, orient=HORIZONTAL)
temperature_scale.set(1.0)
temperature_scale.pack(fill=tk.X, padx=5)

# Buttons frame for submitting and refreshing
button_frame = tk.Frame(root)
button_frame.pack(pady=10, fill=tk.X)
submit_button = tk.Button(button_frame, text="Submit", command=submit_action)
submit_button.pack(side=tk.LEFT, padx=5)
refresh_button = tk.Button(button_frame, text="Refresh", command=reset_and_refresh)
refresh_button.pack(side=tk.LEFT, padx=5)
attach_button = tk.Button(button_frame, text="Select Files", command=select_attachment)
attach_button.pack(side=tk.LEFT, padx=5)

# File label for attachment status
file_label = tk.Label(root, text="No files selected")
file_label.pack(fill=tk.X, pady=5)

# Start the GUI event loop
root.mainloop()