import streamlit as st
import pandas as pd
import torch
from PIL import Image
import re
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load pre-trained models for classification and extraction
# Classification model (DiT)
processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")

# Extraction model (Donut)
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

# Streamlit UI setup with custom theme
st.set_page_config(page_title="Document Classification & Extraction")
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #0B60B0; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Document Classification and Entity Extraction Pipeline")
uploaded_files = st.file_uploader("Upload one or more document images", type=["png", "jpg", "jpeg", "tiff", "tif"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"View and Process {uploaded_file.name}", expanded=True):
            st.image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            # Step 1: Document Classification
            image = Image.open(uploaded_file).convert("RGB").resize((480, 480))  # Resize to standard input size
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits

            # Model predicts one of the 16 RVL-CDIP classes
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]

            # Display classification result
            st.subheader("Classification Result")
            st.write(f"Predicted Document Type: **{predicted_label}**")

            # Add a text input for custom questions
            custom_question = st.text_input(f"Add a custom question for {uploaded_file.name} (e.g., 'What is _______?')")

            # Combine default questions with the custom question if provided
            all_questions = default_questions.copy()
            if custom_question.strip():  # Check if a custom question is provided and not just empty/whitespace
                all_questions.append(custom_question)

            # Select questions to ask for each image
            selected_questions = st.multiselect(
                f"Select questions for {uploaded_file.name}",
                all_questions,
                default=all_questions  # By default, select all questions including the custom one if present
            )

            # Process each uploaded image and extract answers
            if st.button(f"Extract Answers for {uploaded_file.name}"):
                image = Image.open(uploaded_file).convert("RGB")
                results = []

                for question in selected_questions:
                    # Prepare the question prompt and input
                    prompt = task_prompt.replace("{user_input}", question)
                    decoder_input_ids = extraction_processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
                    pixel_values = extraction_processor(image, return_tensors="pt").pixel_values

                    # Generate the answer using the extraction model
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

                    # Extract the answer and append it to the results
                    json_output = extraction_processor.token2json(sequence)
                    answer = json_output.get('answer', None)
                    results.append({"image": uploaded_file.name, "question": question, "answer": answer})

                # Display results as a DataFrame under the image
                if results:
                    df = pd.DataFrame(results)
                    st.write(f"Extracted Entities for {uploaded_file.name}")
                    st.dataframe(df)

                    # Save the DataFrame as a CSV
                    csv_filename = f"{uploaded_file.name}_question_answers.csv"
                    st.download_button(
                        label=f"Download CSV for {uploaded_file.name}",
                        data=df.to_csv(index=False),
                        file_name=csv_filename,
                        mime="text/csv"
                    )

                    # Save the DataFrame as an Excel file using BytesIO
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)  # Move to the beginning of the BytesIO buffer

                    # Provide a download link for the Excel file
                    excel_filename = f"{uploaded_file.name}_question_answers.xlsx"
                    st.download_button(
                        label=f"Download Excel for {uploaded_file.name}",
                        data=excel_buffer,
                        file_name=excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
