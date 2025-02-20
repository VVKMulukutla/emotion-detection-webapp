import streamlit as st
from login_hf import *
from model_callbacks import *

login_into_hf()

def main():
    st.set_page_config(page_title="Facial Emotion Recognition")
    st.title("EmoDet: Determine the facial expression with DL and VLMs")

    with st.form("input_form"):
        image_path = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        option = st.selectbox("Choose an option", ["Microsoft Phi-3.5 Vision Instruct",
                                                   "Meta Llama-3.2 Vision Instruct",
                                                   "PaliGemma2 Vision Instruct",
                                                   "VGG19",
                                                   "ResNet50",
                                                   "EfficientNet50"])
        if option == "Microsoft Phi-3.5 Vision Instruct" or option == "PaliGemma2 Vision Instruct" or option == "Meta Llama-3.2 Vision Instruct":
            user_prompt = st.text_input("Enter your Instruction(Optional), {Works only for VLMs}")
        else:
            user_prompt = ""#my prompt
        submitted = st.form_submit_button("Recognize")

        if submitted:
            if image_path is not None and option:
                image_bytes = image_path.read()
                with st.spinner("Processing..."):
                    if option == "vgg":
                        result = vgg(image_path)

                    elif option == "resnet":
                        result = resnet(image_path)

                    elif option == "efficientnet":
                        result = efficientnet(image_path)

                    elif option == "phi":
                        result = phi_output(image_path, user_prompt)

                    elif option == "pali":
                        result = pali_output(image_path, user_prompt)

                    processed_text = f"Processed: with {option} and image size {len(image_bytes)} bytes"
                    st.subheader("Generated Output")
                    st.text_area("Result", value=processed_text, height=150)
            else:
                st.warning("Please provide all inputs")

if __name__ == "__main__":
    main()