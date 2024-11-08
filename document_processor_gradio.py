import gradio as gr
import pandas as pd
import torch
from PIL import Image
import re
import io
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Set the environment variable to suppress the tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load pre-trained models for classification and extraction
processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
extraction_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
extraction_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
extraction_model.to(device)

# Define the task prompt template and default questions for entity extraction
task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
default_questions = [
    "What is the Phone number?",
    "What is the Name?",
    "What is the Fax number?",
    "What is the Date?"
]

def process_document(uploaded_file, custom_question, selected_questions):
    # Load image and perform classification
    image = Image.open(uploaded_file).convert("RGB").resize((480, 480))
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits

    # Model predicts one of the 16 RVL-CDIP classes
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    # Document classification result
    classification_result = f"Predicted Document Type: {predicted_label}"

    # Prepare the list of questions
    all_questions = default_questions.copy()
    if custom_question.strip():
        all_questions.append(custom_question)
    
    selected_questions = selected_questions or all_questions
    results = []

    # Perform entity extraction
    for question in selected_questions:
        prompt = task_prompt.replace("{user_input}", question)
        decoder_input_ids = extraction_processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = extraction_processor(image, return_tensors="pt").pixel_values

        outputs = extraction_model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=extraction_model.config.max_length,
            pad_token_id=extraction_processor.tokenizer.pad_token_id,
            eos_token_id=extraction_processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[extraction_processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # Decode the output and clean up
        sequence = extraction_processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(extraction_processor.tokenizer.eos_token, "").replace(extraction_processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        # Extract the answer
        json_output = extraction_processor.token2json(sequence)
        answer = json_output.get('answer', None)
        results.append({"Question": question, "Answer": answer})

    # Convert results to DataFrame and prepare for download
    df = pd.DataFrame(results)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_data = excel_buffer.getvalue()

    return classification_result, df, csv_data, excel_data

# Define Gradio interface components
with gr.Blocks() as demo:
    gr.Markdown("<h2>Document Classification and Entity Extraction</h2>")
    
    with gr.Row():
        uploaded_file = gr.Image(label="Upload Document", type="filepath", interactive=True)
        custom_question = gr.Textbox(label="Add Custom Question", placeholder="e.g., What is the address?")
        selected_questions = gr.CheckboxGroup(
            choices=default_questions, 
            label="Select Questions for Extraction", 
            value=default_questions
        )
    
    with gr.Row():
        classify_button = gr.Button("Classify and Extract")
    
    classification_result = gr.Markdown(label="Document Type")
    extraction_results = gr.DataFrame(label="Extracted Entities")
    
    with gr.Row():
        download_csv = gr.File(label="Download CSV")
        download_excel = gr.File(label="Download Excel")
    
    # Trigger classification as soon as the file is uploaded
    uploaded_file.change(
        fn=process_document, 
        inputs=[uploaded_file, custom_question, selected_questions], 
        outputs=[classification_result, extraction_results, download_csv, download_excel]
    )

# Launch the Gradio interface with public link
demo.launch(share=True)
