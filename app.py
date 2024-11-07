import streamlit as st
import pandas as pd
import torch
from PIL import Image
import re
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Load pre-trained models for classification and extraction
# Document Classification Model (DiT)
processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")

# Entity Extraction Model (Donut)
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

# Main UI Setup
st.title("Document Processing Solution")
st.subheader("Automated Document Classification & Entity Extraction")
st.markdown(
    """
    This application enables automatic classification and entity extraction from uploaded document images.
    It’s designed to streamline document handling for business needs. 
    Please upload your documents to begin.
    """
)

uploaded_files = st.file_uploader("Upload Document Images", type=["png", "jpg", "jpeg", "tiff", "tif"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"Process {uploaded_file.name}", expanded=True):
            st.image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            
            # Step 1: Document Classification
            st.markdown("### Step 1: Document Classification")
            image = Image.open(uploaded_file).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]
            st.write(f"**Predicted Document Type:** {predicted_label}")

            # Step 2: Entity Extraction
            st.markdown("### Step 2: Entity Extraction")
            st.markdown("Select or define specific entities to extract from the document.")

            custom_question = st.text_input(f"Custom question for {uploaded_file.name} (e.g., 'What is the Invoice Number?')")
            all_questions = default_questions.copy()
            if custom_question.strip():
                all_questions.append(custom_question)

            selected_questions = st.multiselect(
                f"Select questions for entity extraction from {uploaded_file.name}",
                all_questions,
                default=all_questions
            )

            # Process extraction upon button click
            if st.button(f"Extract Entities for {uploaded_file.name}"):
                results = []
                for question in selected_questions:
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
                    results.append({"Document": uploaded_file.name, "Question": question, "Answer": answer})

                if results:
                    df = pd.DataFrame(results)
                    st.markdown("### Extracted Entities")
                    st.dataframe(df)

                    # Provide options to download results
                    csv_filename = f"{uploaded_file.name}_extracted_entities.csv"
                    st.download_button(
                        label=f"Download CSV for {uploaded_file.name}",
                        data=df.to_csv(index=False),
                        file_name=csv_filename,
                        mime="text/csv"
                    )

                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)
                    excel_filename = f"{uploaded_file.name}_extracted_entities.xlsx"
                    st.download_button(
                        label=f"Download Excel for {uploaded_file.name}",
                        data=excel_buffer,
                        file_name=excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
