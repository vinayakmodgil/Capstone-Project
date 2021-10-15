import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
from PIL import Image
#import capstone_vinayak as csv
import time_series_functions as tsf
import stock_prediction_algorithm as spa

image = Image.open("stock.jpg")

st.image(image, use_column_width=True)
def load_data():
    final_df = pd.read_csv("Final_stock.csv")
    return final_df

final_df = load_data()

def listdata(df):
    sym_list = []
    for row in df["Symbol"]:
       if row not in sym_list:
          sym_list.append(row)
    return sym_list

sym_list = listdata(final_df)

def main():
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration"])

    if page == "Homepage":
        st.title("** Welcome to StockAI**")
        st.header("we deliver.".upper())
        stock = st.selectbox("Choose a stock: ", sym_list)
        model = st.selectbox("Choose a model",["SARIMAX", "LSTM"])
        go = st.button("Go")
        if go:
           for i in sym_list:
              if stock == i:
                if model == "SARIMAX":
                    sarimax = spa.stock_market_prediction(final_df, i, sarimax=1)
                    st.area_chart(sarimax)

        

    elif page == "Exploration":
        st.title("Data Exploration")
        st.dataframe(final_df.head())

# st.markdown("** Overview **")
# st.markdown("This application is a Streamlit dashboard hosted on Heroku that can be used to explore the results from closing Stock Prices from the launch date of a company to the current date")
# age = st.selectbox("Choose your age: ", np.arange(18, 66, 1))
# age1 = st.slider("Choose your age: ", min_value=16, max_value=66, value=35, step=1)

# artists = st.multiselect("Who are your favourite artists?",
# ["Michael Jackson", "Elvis Presley", "Billy Joel", "Madonna"])

if __name__ == "__main__":
    main()