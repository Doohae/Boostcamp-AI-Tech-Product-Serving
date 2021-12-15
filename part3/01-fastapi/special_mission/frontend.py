import io
import os
from pathlib import Path

import requests
from PIL import Image

import streamlit as st
from app.confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

st.set_page_config(layout="wide")

root_password = 'password'


def main():
    st.title("Question Answering Test")
    context_input = st.text_area(
        "Input Context",
        height=250
    )
    query_input = st.text_input("Input Answerable Question")

    if context_input and query_input:

        st.markdown("-------")

        if st.button("Answer"):
            answer = run_mrc(
                model=model,
                tokenizer=tokenizer,
                context=context_input,
                question=query_input)

            st.write("Answer : ", answer)

        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")

        # 기존 stremalit 코드
        # _, y_hat = get_prediction(model, image_bytes)
        # label = config['classes'][y_hat.item()]
        files = [
            ('files', (uploaded_file.name, image_bytes,
                       uploaded_file.type))
        ]
        response = requests.post("http://localhost:8001/order", files=files)
        label = response.json()["products"][0]["result"]
        st.write(f'label is {label}')


@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')
