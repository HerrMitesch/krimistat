import streamlit as st
import pandas as pd
import plotly.express as px

# Load Data (Assumed already cleaned)
@st.cache_data
def load_data():
    df_cases = pd.read_csv(f"Faelle_combined_clean.csv")#, encoding='ISO-8859-1', sep=';')
    df_victims = pd.read_csv(f"Opfer_Tabellen/final_processed_data.csv")#, encoding='ISO-8859-1', sep=';')
    df_perps = pd.read_csv(f"Tatverdaechtige_combined_clean.csv")#, encoding='ISO-8859-1', sep=';')
    return df_cases, df_victims, df_perps

df_cases, df_victims, df_perps = load_data()

# Sidebar Filters
years = sorted(df_cases['Jahr'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("Select Year", years, index=0)

crime_types = df_cases['Straftat'].unique()
selected_crime = st.sidebar.selectbox("Select Crime Type", crime_types)

# Filter Data
df_filtered_cases = df_cases[(df_cases['Jahr'] == selected_year) & (df_cases['Straftat'] == selected_crime)]
df_filtered_victims = df_victims[(df_victims['Jahr'] == selected_year) & (df_victims['Straftat'] == selected_crime)]
df_filtered_perps = df_perps[(df_perps['Jahr'] == selected_year) & (df_perps['Straftat'] == selected_crime)]

# Main Title
st.title("Crime Data Exploration - German BKA")
st.subheader(f"Year: {selected_year} | Crime: {selected_crime}")

# Overview Metrics
st.metric("Total Cases", df_filtered_cases['Faelle'].sum())
st.metric("Total Victims", df_filtered_victims['Oper insgesamt'].sum())
st.metric("Total Perpetrators", df_filtered_perps['gesamt'].sum())

# Charts
fig_cases = px.bar(df_filtered_cases, x='Stadt', y='Faelle', title="Cases per City")
st.plotly_chart(fig_cases)

fig_victims = px.pie(df_filtered_victims, values='Oper insgesamt', names='Fallstatus', title="Victim Distribution")
st.plotly_chart(fig_victims)

fig_perps = px.histogram(df_filtered_perps, x='Erwachsene_gesamt', title="Age Distribution of Perpetrators")
st.plotly_chart(fig_perps)

# Raw Data
st.subheader("Raw Data")
st.write("Filtered Cases")
st.dataframe(df_filtered_cases)
st.write("Filtered Victims")
st.dataframe(df_filtered_victims)
st.write("Filtered Perpetrators")
st.dataframe(df_filtered_perps)