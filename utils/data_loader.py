import os
import pandas as pd
import gdown
import streamlit as st

FILE_ID = "18vFxh2HBkXdEG5D9I4PmQ2YgPYvHNG75"

@st.cache_data
def load_data():

    csv_path = "creditcard.csv"

    if not os.path.exists(csv_path):

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        gdown.download(
            url,
            csv_path,
            quiet=False
        )

    return pd.read_csv(csv_path)