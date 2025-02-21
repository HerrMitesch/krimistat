import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from data import *
st.set_page_config(layout="wide")

df_cases, df_victims, df_perps = load_data()
df_dashboard = load_dashboard_data()

from content import *

def create_overview_page():
    st.title("krimistat")
    st.markdown("### Eine Visualisierung der Kriminalitätsstatistik des BKA")
    st.subheader("Zusammenfassung Metriken")
    st.write(
        """
        Willkommen bei krimistat. Nutzen Sie die Seitenleiste, um zwischen den verschiedenen Seiten zu navigieren. 
        Entdecken Sie die Gesamtdaten auf der Übersichtsseite, detaillierte Fallinformationen auf der Fälle-Seite, 
        Opferdetails auf der Opfer-Seite und schließlich Täterdetails auf der Täter-Seite.
        """
    )
    total_cases = df_cases['Faelle'].sum() if 'Faelle' in df_cases.columns else "N/A"
    total_victims = df_victims['Oper insgesamt'].sum() if 'Oper insgesamt' in df_victims.columns else "N/A"
    total_perpetratrors = df_perps['gesamt'].sum() if 'gesamt' in df_perps.columns else "N/A"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gesamtzahl der Fälle", total_cases)
    with col2:
        st.metric("Gesamtzahl der Opfer", total_victims)
    with col3:
        st.metric("Gesamtzahl der Täter", total_perpetratrors)
    st.image(r"Opfer_Tabell\invest.webp", width=600)

    
    
     
def create_data_page():
    st.title("📊 Rohdaten")
    st.markdown("### Rohdaten zu Fällen, Opfern und Tatverdächtigen")
    st.divider()
    
    st.subheader("Fälle")
    st.dataframe(df_cases)
    st.divider()
    
    st.subheader("Opfer")
    st.dataframe(df_victims)
    st.divider()

    st.subheader("Tatverdächtige")
    st.dataframe(df_perps)


def create_cases_page():
    st.title("🔫 Fälle")
    st.markdown("### Visualisierung der Falldaten")
    st.divider()

    years = sorted(df_cases['Jahr'].unique(), reverse=True)
    crime_types = df_cases['Vereinfachte_Straftat'].unique()

    st.sidebar.title("Optionen")
    selected_year = st.sidebar.slider("Jahr auswählen:", min_value=2016, max_value = 2023, value = 2020)
    
    selected_crime = st.selectbox("Straftat auswählen", crime_types)
    
    df_filtered_cases = df_cases[(df_cases['Jahr'] == selected_year) & (df_cases['Vereinfachte_Straftat'] == selected_crime)]
    
    st.metric("Gesamtzahl der Fälle", df_filtered_cases['Faelle'].sum())
    
    
    
    # Load Data (Assuming df_dashboard is already loaded)
    st.title("Dashboard")
    

    
    
    if selected_crime != "Alle":
        df_filtered = df_cases[df_cases["Straftat"] == selected_crime]
    else:
        df_filtered=df_cases
    #######
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    
    
    ######
    # Layout: Side-by-Side Charts (Map + Pie Chart)
    c1, c2 = st.columns(2)

    with c1:
        #st.subheader("📍 Kriminalitätskarte")
        city_coords = get_city_coordinates()
        df_cases_with_coords = df_filtered_cases.merge(city_coords, on='Stadt', how='left')
        fig_map = create_map(df_cases_with_coords)
        st.plotly_chart(fig_map, use_container_width=True)

    with c2:
        #st.subheader("🔝 Top 10 Kriminalitätsarten")
        filtered_df = df_cases[(df_cases['Jahr'] == selected_year) & ~df_cases['Vereinfachte_Straftat'].str.contains("insgesamt", case=False, na=False)]
        top_crimes = filtered_df.groupby("Vereinfachte_Straftat")["Faelle"].sum().nlargest(10)

        if not top_crimes.empty:
            fig_pie = px.pie(top_crimes, values=top_crimes.values, names=top_crimes.index, title="Top 10 Häufigste Verbrechen")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:

            st.warning(f"Keine Kriminalitätsdaten für {selected_year} verfügbar.")   

              

        # Remove "Straftaten insgesamt" from data
        df_filtered = df_cases[~df_cases["Vereinfachte_Straftat"].str.contains("insgesamt", case=False, na=False)]

        # Apply year filter
        # df_filtered = df_filtered[df_filtered["Jahr"] == selected_year]

        # Summarize cases per crime type
        df_total_per_crime = df_filtered.groupby("Vereinfachte_Straftat")["Faelle"].sum().reset_index()

        # Get the 10 most frequent crimes
        top_10_crimes = df_total_per_crime.nlargest(10, "Faelle")["Vereinfachte_Straftat"]

        # Filter data for only top 10 crimes
        df_top_crimes = df_filtered[df_filtered["Vereinfachte_Straftat"].isin(top_10_crimes)]

        # Group by year and crime type
        df_grouped = df_top_crimes.groupby(["Jahr", "Vereinfachte_Straftat"])["Faelle"].sum().reset_index()

        # Create interactive line plot
        fig = px.line(df_grouped, 
              x="Jahr", 
              y="Faelle", 
              color="Vereinfachte_Straftat",
              markers=True,
              title=f"Top 10 Verbrechen zwischen 2016 und 2023",
              labels={"Jahr": "Jahr", "Faelle": "Anzahl Fälle", "Vereinfachte_Straftat": "Straftat"})

        # Show the plot in Streamlit
        st.plotly_chart(fig)



    st.divider()


    #City filter
    cities = ["Alle Städte"] + sorted(df_cases["Stadt"].unique().tolist())
    selected_city = st.selectbox("Stadt auswählen:", cities, index=0)
    display_mode = st.selectbox("Darstellungsform:", ["absolut","relativ"], index=0)

    # Filter data based on selected city
    if selected_city == "Alle Städte":
        df_filtered = df_cases  # Use full dataset
    else:
        df_filtered = df_cases[(df_cases["Jahr"] == selected_year)&(df_cases["Stadt"] == selected_city)& ~df_cases['Vereinfachte_Straftat'].str.contains("insgesamt", case=False, na=False)]
    years = sorted(df_cases['Jahr'].unique(), reverse=True)
    crime_types = df_cases['Vereinfachte_Straftat'].unique()

    if display_mode == "absolut":
        plot_gender_distribution(df_filtered, selected_city,selected_year)
    elif display_mode == "relativ":
        plot_gender_fraction(df_filtered, selected_city,selected_year)
    
    st.divider()

def create_victims_page():
    st.title("😢 Opfer")
    st.markdown("### Visualisierung der Opferdaten")
    st.divider()
    years = sorted(df_victims['Jahr'].unique(), reverse=True)
    crime_types = df_victims['Straftat'].unique()
    
    st.sidebar.title("Optionen")
    selected_year = st.sidebar.slider("Jahr auswählen", min_value=2016, max_value = 2023, value = 2020)
    
    #selected_year = st.selectbox("Select Year", years, index=0)
    selected_crime = st.selectbox("Wählen Sie die Art der Straftat aus", crime_types)
    
    df_filtered_victims = df_victims[(df_victims['Jahr'] == selected_year) & (df_victims['Straftat'] == selected_crime)]
    
    st.metric("Gesamtzahl der Opfer", df_filtered_victims['Oper insgesamt'].sum())
    
    # Load Data (Assuming df_dashboard is already loaded)
    st.title("Dashboard")

    # City Filter
    city_list = ["Alle"] + sorted(df_dashboard["Stadt"].unique().tolist())
    selected_city = st.sidebar.selectbox("Wählen Sie die Stadt aus", city_list)
    
    #Crime Type Filter
    #selected_crime = st.selectbox("Select Crime Type", crime_types)
    crime_list=["Alle"] + sorted(df_dashboard["Straftat"].unique().tolist())
    selected_crime= st.sidebar.selectbox("Wählen Sie die Art der Straftat aus",crime_list)

    # Filter Data
    if selected_city != "Alle":
        
        df_filtered = df_dashboard[df_dashboard["Stadt"] == selected_city]
    else:
        df_filtered = df_dashboard
    
    if selected_crime != "Alle":
        df_filtered = df_filtered[df_filtered["Straftat"] == selected_crime]
    else:
        df_filtered=df_filtered
    

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Crime Trends Over the Years
    crime_trend = df_filtered.groupby("Jahr")["Oper insgesamt"].sum()
    sns.lineplot(x=crime_trend.index, y=crime_trend.values, marker="o", ax=axes[0, 0])
    axes[0, 0].set_title("Kriminalitätstrends (2016–2023)")
    axes[0, 0].set_xlabel("Jahr")
    axes[0, 0].set_ylabel("Gesamtzahl Opfer")

    # Male vs Female Victim Trends
    crime_trend_male = df_filtered.groupby("Jahr")["Opfer maennlich"].sum()
    crime_trend_female = df_filtered.groupby("Jahr")["Opfer weiblich"].sum()

    sns.lineplot(x=crime_trend_male.index, y=crime_trend_male.values, marker="o", label="Männlich", ax=axes[0, 1])
    sns.lineplot(x=crime_trend_female.index, y=crime_trend_female.values, marker="s", label="Weiblich", ax=axes[0, 1])
    axes[0, 1].set_title("Trends bei männlichen vs. weiblichen Opfern (2016–2023)")
    axes[0, 1].set_xlabel("Jahr")
    axes[0, 1].set_ylabel("Gesamtzahl Opfer")
    axes[0, 1].legend()

    # Crime Heatmap (Top 20 Cities)
    top_20_cities = df_dashboard.groupby("Stadt")["Oper insgesamt"].sum().nlargest(20).index
    df_filtered_top20 = df_dashboard[df_dashboard["Stadt"].isin(top_20_cities)]
    df_pivot = df_filtered_top20.pivot_table(values="Oper insgesamt", index="Stadt", columns="Jahr", aggfunc="sum", fill_value=0)
    sns.heatmap(df_pivot, cmap="Blues", linewidths=0.5, ax=axes[1, 0])

    axes[1, 0].set_title("Opfer nach Stadt im Zeitverlauf")

    # Victim Distribution by Age Category
    age_categories = {
    "Opfer - Kinder bis unter 6 Jahre - insgesamt": "0-5 Jahre",
    "Opfer Kinder 6 bis unter 14 Jahre - insgesamt": "6-13 Jahre",
    "Opfer Jugendliche 14 bis unter 18 Jahre - insgesamt": "14-17 Jahre",
    "Opfer - Heranwachsende 18 bis unter 21 Jahre - insgesamt": "18-20 Jahre",
    "Opfer Erwachsene 21 bis unter 60 Jahre - insgesamt": "21-59 Jahre",
    "Opfer - Erwachsene 60 Jahre und aelter - insgesamt": "60+ Jahre"
}
    df_age_victims = df_filtered[list(age_categories.keys())].sum().rename(index=age_categories)
    sns.barplot(x=df_age_victims.index, y=df_age_victims.values, palette="Blues", ax=axes[1, 1])

    axes[1, 1].set_title("Opferverteilung nach Alterskategorie")
    axes[1, 1].set_xlabel("Alterskategorie")
    axes[1, 1].set_ylabel("Gesamtzahl der Opfer")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

    # Display plots
    st.pyplot(fig)

def create_perpetrators_page():
    st.title("🦹‍♂️ Tatverdächtige")
    st.markdown("### Visualisierung der Daten zu Tatverdächtigen")
    st.divider()


    
    # ---- FILTERS ----
    years = sorted(df_perps["Jahr"].unique(), reverse=True)
    st.sidebar.title("Optionen")
    selected_year = st.sidebar.slider("Jahr auswählen", min_value=2016, max_value = 2023, value = 2020)

    cities = ["Alle Städte"] + sorted(df_perps["Stadt"].dropna().unique())
    selected_city = st.sidebar.selectbox("Stadt auswählen", cities, index=0)

    # Apply filters
    df_filtered = df_perps[df_perps["Jahr"] == selected_year]
    if selected_city != "Alle Städte":
        df_filtered = df_filtered[df_filtered["Stadt"] == selected_city]


    # ---- 3️⃣ TOP CRIMES BY GENDER ----
    selected_crime = st.sidebar.selectbox("Verbrechen auswählen", df_filtered["Vereinfachte_Straftat"].unique())

    # ----  Histogram Toggle ----
    histogram_type = st.toggle("Histogramm reskalieren", value=False)  

    # Define age bins
    age_bins = {
        "Alter_unter_6": (0, 6),
        "Alter_6_bis_8": (6, 8),
        "Alter_8_bis_10": (8, 10),
        "Alter_10_bis_12": (10, 12),
        "Alter_12_bis_14": (12, 14),
        "Alter_14_bis_16": (14, 16),
        "Alter_16_bis_18": (16, 18),
        "Alter_18_bis_21": (18, 21),
        "Alter_21_bis_25": (21, 25),
        "Alter_25_bis_30": (25, 30),
        "Alter_30_bis_40": (30, 40),
        "Alter_40_bis_50": (40, 50),
        "Alter_50_bis_60": (50, 60),
        "Alter_ueber_60": (60, 80)  # Assume max age is 80 for visualization
    }

    # Filter dataset based on selected crime type
    df_filtered_crime = df_filtered[df_filtered["Vereinfachte_Straftat"] == selected_crime]

    # Prepare age data
    age_data = []
    for age_group, (start_age, end_age) in age_bins.items():
        total_cases = df_filtered_crime[age_group].sum()
        age_data.append({"Age Group": age_group, "Start Age": start_age, "End Age": end_age, "Cases": total_cases})

    df_age = pd.DataFrame(age_data)

    # ---- Create Histograms ----
    if histogram_type:
        fig = go.Figure()
        for _, row in df_age.iterrows():
            fig.add_trace(go.Bar(
                x=[(row["Start Age"] + row["End Age"]) / 2],  # Center the bar
                y=[row["Cases"] / (row["End Age"] - row["Start Age"])],  # Normalize for width
                width=[row["End Age"] - row["Start Age"]],  # Proper width
                text=[f"{row['Cases']} "],
                textposition="outside",
                name=f"{row['Start Age']} - {row['End Age']}"
            ))

        fig.update_layout(
            title=f"Altersverteilung der Tatverdächtigen (reskaliert) - {selected_crime}",
            xaxis_title="Alter",
            yaxis_title="Fälle pro Jahr",
            xaxis=dict(tickmode="linear", dtick=2),  # Add ticks every 2 years
            showlegend=False
        )
    else:
        fig = px.bar(df_age, x="Age Group", y="Cases", 
                    labels={"Age Group":"Altersgruppe",
                    "Cases":"Fälle pro Jahr"},
                    title=f"Altersverteilung der Tatverdächtigen - {selected_crime}",
                    text_auto=True)

    st.plotly_chart(fig)

def create_regression_page():
    st.title("📈 Verbrechen vs. Einwohner")
    st.markdown("### Zusammenhang zwischen Einwohnerzahl und Anzahl der Verbrechen")
    st.divider()
    plot_crimes_vs_inhabitants(df_cases)


# Sidebar Navigation
st.sidebar.title("Navigation")



page = st.sidebar.radio("geh zu", ["Übersicht", "Rohdaten", "Fälle", "Opfer", "Täter","Regression"])



# Page Selection
if page == "Übersicht":
    create_overview_page()
elif page == "Rohdaten":
    create_data_page()
elif page == "Fälle":
    create_cases_page()
elif page == "Opfer":
    create_victims_page()
elif page == "Täter":
    create_perpetrators_page()
elif page == "Regression": 
    create_regression_page()