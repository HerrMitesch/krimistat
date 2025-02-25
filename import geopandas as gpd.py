import streamlit as st

# Set page configuration (this must come first)
st.set_page_config(page_title="Crime Data Dashboard", layout="wide")

# Inject custom CSS for background image, colors, etc.
st.markdown(
    """
    <style>
        .stApp {
            background: url("https://source.unsplash.com/1600x900/?city,crime") no-repeat center center fixed;
            background-size: cover;
        }
        .main {
            background-color: rgba(245, 245, 245, 0.9); /* semi-transparent background */
            padding: 1rem;
            border-radius: 8px;
        }
        h1, h2, h3 {
            color: #1E3A8A;
        }
        .sidebar .sidebar-content {
            background-color: #FFFFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
import pandas as pd

@st.cache_data
def load_data():
    df_cases = pd.read_csv("Faelle_combined_clean.csv")
    # Adjust file paths as needed
    df_victims = pd.read_csv("Opfer_Tabell/Opfer_combined_clean_2016_2023.csv", encoding='ISO-8859-1', sep=';')
    return df_cases, df_victims

df_cases, df_victims = load_data()

# If needed for dashboard visualizations:
@st.cache_data
def load_dashboard_data():
    # Make sure the file path is correct; adjust if needed
    return pd.read_csv("Opfer_combined_clean_2016_2023.csv", encoding='ISO-8859-1', sep=';')

df_dashboard = load_dashboard_data()

import matplotlib.pyplot as plt
import seaborn as sns

def create_dashboard_page():
    st.title("📊 Crime Data Dashboard")
    
    # Optionally, display a header image
    st.image(r"C:\Users\PC\Desktop\Projekt_Weiterbildung\krimistat\Opfer_Tabell\invest.webp", use_column_width=True)
    
    # Sidebar filters specific to dashboard (optional)
    st.sidebar.header("Dashboard Filters")
    selected_year = st.sidebar.selectbox("Select Year", sorted(df_dashboard["Jahr"].unique()), index=len(df_dashboard["Jahr"].unique())-1)
    selected_city = st.sidebar.multiselect("Select City", df_dashboard["Stadt"].unique())
    selected_crime = st.sidebar.multiselect("Select Crime Type", df_dashboard["Straftat"].unique())
    
    # Apply filters to dashboard data
    df_filtered = df_dashboard[df_dashboard["Jahr"] == selected_year]
    if selected_city:
        df_filtered = df_filtered[df_filtered["Stadt"].isin(selected_city)]
    if selected_crime:
        df_filtered = df_filtered[df_filtered["Straftat"].isin(selected_crime)]
    
    st.write("### Summary Statistics")
    st.dataframe(df_filtered.describe())
    
    # Crime Trends Over the Years
    st.write("### Crime Trends Over the Years")
    crime_trend = df_dashboard.groupby("Jahr")["Oper insgesamt"].sum()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=crime_trend.index, y=crime_trend.values, marker="o", ax=ax1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Crimes")
    ax1.set_title("Crime Trends (2014-2023)")
    st.pyplot(fig1)
    
    # Top 10 Crimes (Bar Chart)
    st.write("### Top 10 Most Reported Crimes")
    top_crimes = df_filtered.groupby("Straftat")["Oper insgesamt"].sum().nlargest(10)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="viridis", ax=ax2)
    ax2.set_xlabel("Number of Cases")
    ax2.set_ylabel("Crime Type")
    ax2.set_title("Top 10 Crimes")
    st.pyplot(fig2)
    
    # Male vs Female Victim Trends
    st.write("### Male vs Female Victims Over Time")
    crime_trend_male = df_dashboard.groupby("Jahr")["Opfer maennlich"].sum()
    crime_trend_female = df_dashboard.groupby("Jahr")["Opfer weiblich"].sum()
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=crime_trend_male.index, y=crime_trend_male.values, marker="o", label="Male Victims", ax=ax3)
    sns.lineplot(x=crime_trend_female.index, y=crime_trend_female.values, marker="s", label="Female Victims", ax=ax3)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Total Victims")
    ax3.legend()
    ax3.set_title("Male vs Female Victim Trends (2014-2023)")
    st.pyplot(fig3)
    
    # Heatmap for Crime by City
    st.write("### Crime Heatmap (Top 20 Cities)")
    top_20_cities = df_filtered.groupby("Stadt")["Oper insgesamt"].sum().nlargest(20).index
    df_filtered_top20 = df_filtered[df_filtered["Stadt"].isin(top_20_cities)]
    df_pivot = df_filtered_top20.pivot_table(values="Oper insgesamt", index="Stadt", columns="Jahr", aggfunc="sum", fill_value=0)
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_pivot, cmap="Blues", linewidths=0.5, ax=ax4)
    ax4.set_title("Victims by City Over Time")
    st.pyplot(fig4)
    
    # Download Data Button in Sidebar (if desired)
    st.sidebar.write("## Download Processed Data")
    st.sidebar.download_button(
        label="Download CSV",
        data=df_dashboard.to_csv(index=False),
        file_name="processed_crime_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.markdown("📌 **Crime Data Dashboard - Built with Streamlit**")
# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Übersicht", "Fälle", "Opfer", "Dashboard"])

if page == "Übersicht":
    create_overview_page()
elif page == "Fälle":
    create_cases_page()
elif page == "Opfer":
    create_victims_page()
elif page == "Dashboard":
    create_dashboard_page()