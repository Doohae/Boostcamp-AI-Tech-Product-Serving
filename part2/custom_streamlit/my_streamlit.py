# "streamlit run my_streamlit.py" to run
import streamlit as st

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import tokenizers

from mrc import run_mrc
from confirm_button_hack import cache_on_button_press

st.title('Question Answering Machine Test')

root_password = 'password'

# https://stackoverflow.com/questions/70274841/streamlit-unhashable-typeerror-when-i-use-st-cache/70275957
@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None})
def load_model(path):
    model = AutoModelForQuestionAnswering.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password


# https://huggingface.co/ainize/klue-bert-base-mrc
data_load_state = st.text('Loading model and tokenizer...')
model, tokenizer = load_model("ainize/klue-bert-base-mrc")
data_load_state.text("Model & Tokenizer Uploaded!")

st.markdown("---------")

st.subheader("Input Your Own Text Here!")

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

st.markdown("---------")

st.subheader("Authentifiaction required to see model info")
if st.checkbox("Show Model URL"):
    st.write("https://huggingface.co/ainize/klue-bert-base-mrc")
password = st.text_input('Input Password Below', type="password")
if authenticate(password):
    st.success('You are authenticated!')
    st.write(model)
else:
    st.error('Invalid Password')