import gradio as gr
import pandas as pd
import torch
from PIL import Image
import re
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load pre-trained models for classification and extraction
processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")

extraction_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
extraction_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
extraction_model.to(device)

# Define prompt for entity extraction
task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
default_questions = [
    "What is the Phone number?",
    "What is the Name?",
    "What is the Fax number?",
    "What is Date?"
]

def classify_and_extract(images, custom_question):
    all_results = []
    
    for uploaded_image in images:
        # Classification Step
        image = Image.open(uploaded_image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        # Entity Extraction Step
        extraction_results = []
        all_questions = default_questions.copy()
        if custom_question.strip():
            all_questions.append(custom_question)

        for question in all_questions:
            prompt = task_prompt.replace("{user_input}", question)
            decoder_input_ids = extraction_processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
            pixel_values = extraction_processor(image, return_tensors="pt").pixel_values

            outputs = extraction_model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                max_length=extraction_model.decoder.config.max_position_embeddings,
                pad_token_id=extraction_processor.tokenizer.pad_token_id,
                eos_token_id=extraction_processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[extraction_processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            sequence = extraction_processor.batch_decode(outputs.sequences)[0]
            sequence = re.sub(r"<.*?>", "", sequence.replace(extraction_processor.tokenizer.eos_token, "").replace(extraction_processor.tokenizer.pad_token, "")).strip()
            json_output = extraction_processor.token2json(sequence)
            answer = json_output.get('answer', None)
            extraction_results.append({"Question": question, "Answer": answer})

        # Store results for each image
        all_results.append({
            "Image": uploaded_image.name,
            "Predicted Document Type": predicted_label,
            "Extracted Entities": extraction_results
        })

    # Prepare output data as a DataFrame
    results_df = pd.DataFrame([
        {
            "Image": res["Image"],
            "Predicted Document Type": res["Predicted Document Type"],
            **{entity["Question"]: entity["Answer"] for entity in res["Extracted Entities"]}
        }
        for res in all_results
    ])

    # Save to CSV and Excel for download
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    excel_buffer = io.BytesIO()
    results_df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)

    return all_results, csv_data, excel_buffer

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Document Processing Solution")
    gr.Markdown("Upload multiple document images for automated classification and entity extraction.")

    image_input = gr.File(label="Upload Document Images", type="filepath", file_count="multiple")
    custom_question_input = gr.Textbox(label="Custom question (e.g., 'What is the Invoice Number?')")

    # Output Components
    results_output = gr.Dataframe(label="Classification and Entity Extraction Results")
    csv_output = gr.File(label="Download Results as CSV")
    excel_output = gr.File(label="Download Results as Excel")

    # Submit Button and Interface
    submit_button = gr.Button("Process")
    submit_button.click(
        fn=classify_and_extract,
        inputs=[image_input, custom_question_input],
        outputs=[results_output, csv_output, excel_output],
    )

demo.launch()
