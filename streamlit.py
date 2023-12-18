import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks import get_openai_callback
from mcq_generator.MCQGEN import generate_evaluate_chain
from src.mcq_generator.util import read_file, get_table_data
from mcq_generator.logger import logging

with open('C:\\Users\\user\\OneDrive\\Desktop\\mcqgen2\\response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

st.title("MCQ APPLICATION")
uploaded_file = st.file_uploader("Upload a PDF or txt file")
topic = st.text_input("Topic name", max_chars=20)
mcq_count = st.number_input("No. of questions", min_value=3, max_value=50)
tone = st.text_input("Level Of Questions", max_chars=20, placeholder="Simple")
button = st.form_submit_button("Create MCQs")

if button and uploaded_file is not None and mcq_count and topic and tone:
    with st.spinner("loading..."):
        try:
            text = read_file(uploaded_file)
            with get_openai_callback() as cb:
                response = generate_evaluate_chain(
                    {
                        "text": text,
                        "topic": topic,
                        "number": mcq_count,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON),
                    }
                )
        except Exception as e:
            traceback.print_exception(type(e), e, e._traceback__)
            st.error("Error")
        else:
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost: {cb.total_cost}")
            
            if isinstance(response, dict):
                quiz = response.get("quiz", None)
                if quiz is not None:
                    table_data = get_table_data(quiz)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1
                        st.table(df)
                        st.text_area(label="Review", value=response["review"])
                    else:
                        st.error("Error in the table data")
                else:
                    st.write(response)
