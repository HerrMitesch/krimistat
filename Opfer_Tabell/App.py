import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page title

st.set_page_config(page_title="Crime Data Dashboard", layout="wide")


# Inject custom CSS for background color and styling
st.markdown(
    """
    <style>
        .main {
            background-color: #F5F5F5;
        }
        h1, h2, h3 {
            color: #1E3A8A; /* Red color for titles */
        }
        .sidebar .sidebar-content {
            background-color: #FFFFFF !important;
        }
        .stApp {
            background: url("https://source.unsplash.com/1600x900/?city,crime") no-repeat center center fixed;
            background-size: cover;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Opfer_combined_clean_2016_2023.csv", encoding='ISO-8859-1', sep=';')
    return df

df = load_data()

# Display Image Header
#st.image("https://source.unsplash.com/1600x900/?crime", use_container_width=True)
st.image(r"C:\Users\PC\Desktop\Projekt_Weiterbildung\krimistat\Opfer_Tabell\invest.webp",  use_container_width=True)# Sidebar Filters
st.sidebar.image("https://source.unsplash.com/200x200/?logo", width=200)
st.sidebar.header("Filter Data")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["Jahr"].unique()), index=len(df["Jahr"].unique())-1)
selected_city = st.sidebar.multiselect("Select City", df["Stadt"].unique())
selected_crimetype=st.sidebar.multiselect("Select crime_type",df["Straftat"].unique())

# Apply Filters
df_filtered = df[df["Jahr"] == selected_year]
if selected_city:
    df_filtered = df_filtered[df_filtered["Stadt"].isin(selected_city)]
if selected_crimetype:
    df_filtered = df_filtered[df_filtered["Straftat"].isin(selected_crimetype)]

# Display Data
st.title("📊 Crime Data Dashboard")
st.write("### Summary Statistics")
st.dataframe(df_filtered.describe())

# Crime Trends
st.write("### Crime Trends Over the Years")
crime_trend = df.groupby("Jahr")["Oper insgesamt"].sum()
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=crime_trend.index, y=crime_trend.values, marker="o", ax=ax)
plt.xlabel("Year")
plt.ylabel("Total Crimes")
plt.title("Crime Trends (2014-2023)")
st.pyplot(fig)

# Crime by Type (Bar Chart)
st.write("### Top 10 Most Reported Crimes")
top_crimes = df_filtered.groupby("Straftat")["Oper insgesamt"].sum().nlargest(10)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="viridis", ax=ax)
plt.xlabel("Number of Cases")
plt.ylabel("Crime Type")
plt.title("Top 10 Crimes")
st.pyplot(fig)

# Victim Trends by Gender
st.write("### Male vs Female Victims Over Time")
crime_trend_male = df.groupby("Jahr")["Opfer maennlich"].sum()
crime_trend_female = df.groupby("Jahr")["Opfer weiblich"].sum()

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=crime_trend_male.index, y=crime_trend_male.values, marker="o", label="Male Victims", ax=ax)
sns.lineplot(x=crime_trend_female.index, y=crime_trend_female.values, marker="s", label="Female Victims", ax=ax)
plt.xlabel("Year")
plt.ylabel("Total Victims")
plt.legend()
plt.title("Male vs Female Victim Trends (2014-2023)")
st.pyplot(fig)

# Heatmap for Crime by City
st.write("### Crime Heatmap (Top 20 Cities)")
top_20_cities = df_filtered.groupby("Stadt")["Oper insgesamt"].sum().nlargest(20).index
df_filtered_top20 = df_filtered[df_filtered["Stadt"].isin(top_20_cities)]
df_pivot = df_filtered_top20.pivot_table(values="Oper insgesamt", index="Stadt", columns="Jahr", aggfunc="sum", fill_value=0)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_pivot, cmap="Blues", linewidths=0.5, ax=ax)
plt.title("Victims by City Over Time")
st.pyplot(fig)

# Export Data Button
st.sidebar.write("## Download Processed Data")
st.sidebar.download_button(label="Download CSV", data=df.to_csv(index=False), file_name="processed_crime_data.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("📌 **Crime Data Dashboard - Built with Streamlit**")