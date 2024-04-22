import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

model = pickle.load(open('kmeans_model.pkl', 'rb'))

def ScaleAnnualIncome(annual_income):
    return annual_income/1000

def predict_cluster(annual_income, spending_score):
    scaled_annual_income = ScaleAnnualIncome(annual_income)
    data = np.array([[scaled_annual_income, spending_score]])
    cluster = model.predict(data)
    return cluster[0]

st.title("Clustering by Annual Income and Spending Score")
input_annualIncome = st.text_input("Enter annual income ($)")
input_spendingScore = st.text_input("Enter spending score (1-100)")

if(st.button("Predict Cluster")):
    try:
        annual_income = float(input_annualIncome)
        spending_score = float(input_spendingScore)
        #annual_income = ScaleAnnualIncome(annual_income)
        cluster = predict_cluster(annual_income, spending_score)
        st.write(f"This customer belongs to Cluster {cluster+1}\n")
        if cluster == 0:
            st.write("Customer has a middling annual income and spending score. They are a potential target customer.")
        elif cluster == 1:
            st.write("Customer has high income and spending score. Continue targeting this customer.")
        elif cluster == 2:
            st.write("Customer has high annual income but low spending score. Consider appealing to their needs and wants")
        elif cluster == 3:
            st.write("Customer has low annual income and low spending score.")
        elif cluster == 4:
            st.write("Customer has low income but high spending score. Continue targeting this customer.")
    except ValueError:
        st.error("Please enter valid numerical values for the Spending Score and Annual Income")