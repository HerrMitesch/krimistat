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

# Function to create the map
def create_map(df_filtered_cases):
    fig = px.scatter_mapbox(df_filtered_cases, lat='lat', lon='lon', size='Faelle', hover_name='Stadt',
                            title="Straftaten Karte", mapbox_style="carto-positron", zoom=4.9, height=820, width=1000)
    
    fig.update_layout(clickmode='event+select')  # Enable click events
    return fig

def plot_crimes_vs_inhabitants(df_cases):
    df_cases_filtered = df_cases[(df_cases["HZ"] > 0) & (df_cases['Straftat'] == "Straftaten insgesamt")]  # Avoid division by zero
    df_cases_filtered["Inhabitants"] = (df_cases_filtered["Faelle"] * 100000) / df_cases_filtered["HZ"]

    fig = px.scatter(df_cases_filtered, x="Inhabitants", y="Faelle", hover_name="Stadt",
                     title="Crimes vs. Inhabitants per City",
                     labels={"Inhabitants": "Number of Inhabitants", "Faelle": "Number of Crimes"})

    st.plotly_chart(fig, use_container_width=True)    

def plot_crimes_vs_inhabitants(df_cases):
    # Filter out rows where "HZ" is 0 to avoid division by zero
    df_cases_filtered = df_cases[(df_cases["HZ"] > 0) & (df_cases["Straftat"]=="Straftaten insgesamt")]  
    df_cases_filtered["Inhabitants"] = (df_cases_filtered["Faelle"] * 100000) / df_cases_filtered["HZ"]

    # Create the scatter plot
    fig = px.scatter(df_cases_filtered, x="Inhabitants", y="Faelle", hover_name="Stadt",
                     title="Verbrechenszahl abhängig von Stadtgröße",
                     labels={"Inhabitants": "Anzahl der Einwohner", "Faelle": "Anzahl der Verbrechen"})
    
    show_fit = st.toggle("Fit anzeigen", value=False)

    if show_fit:
        # Prepare the data for training
        X = df_cases_filtered["Inhabitants"].values.reshape(-1, 1)  # Features (Inhabitants)
        y = df_cases_filtered["Faelle"].values  # Target variable (Number of Crimes)
    
        # Split data into 90% training and 10% validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        # Initialize the linear regression model
        model = LinearRegression()

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the validation data
        y_pred = model.predict(X_val)

        # Calculate the Mean Squared Error (MSE) on the validation data
        mse = mean_squared_error(y_val, y_pred)

        

        # Plot the regression line along with the data points
        fig_regression = px.scatter(df_cases_filtered, x="Inhabitants", y="Faelle", hover_name="Stadt",
                                title="Verbrechenszahl abhängig von Stadtgröße",
                                labels={"Inhabitants": "Anzahl der Einwohner", "Faelle": "Anzahl der Verbrechen"})
        fig_regression.add_scatter(x=df_cases_filtered["Inhabitants"], 
                               y=model.predict(df_cases_filtered["Inhabitants"].values.reshape(-1, 1)), 
                               mode='lines', name="Regressionslinie", line=dict(color='red'))
        st.plotly_chart(fig_regression, use_container_width=True)
        # Display the results
        #st.write(f"Mean Squared Error (MSE) on validation data: {mse:.2f}")
        #st.write(f"Regression Coefficients: Intercept = {model.intercept_:.2f}, Slope = {model.coef_[0]:.2f}")
    else:
        st.plotly_chart(fig, use_container_width=True)
    


def plot_gender_distribution(df_filtered, selected_city, selected_year):
    """Plots absolute gender distribution for different crimes."""
    
    if df_filtered.empty:
        st.warning(f"No data available for {selected_city} in {selected_year}.")
        return

    # Aggregate data if "All Cities" is selected
    if selected_city == "All Cities":
        df_filtered = df_filtered.groupby("Vereinfachte_Straftat", as_index=False)[["Tatverdaechtige_maennlich", "Tatverdaechtige_weiblich"]].sum()

    # Compute gender imbalance (absolute difference)
    df_filtered["Imbalance"] = abs(df_filtered["Tatverdaechtige_maennlich"] - df_filtered["Tatverdaechtige_weiblich"])

    # Sort by gender imbalance (largest difference first)
    df_filtered = df_filtered.sort_values(by="Imbalance", ascending=False)

    # Convert to long format for grouped bar plot
    df_long = df_filtered.melt(id_vars=["Vereinfachte_Straftat"], 
                               value_vars=["Tatverdaechtige_maennlich", "Tatverdaechtige_weiblich"],
                               var_name="Gender", value_name="Number")

    # Rename gender labels
    df_long["Gender"] = df_long["Gender"].replace({
        "Tatverdaechtige_maennlich": "Männlich ",
        "Tatverdaechtige_weiblich": "Weiblich"
    })

    # Create grouped bar chart
    fig = px.bar(df_long, 
                 x="Vereinfachte_Straftat", 
                 y="Number", 
                 color="Gender",
                 barmode="group", 
                 title=f"Geschlechterverteilung der Tatverächtigen in {selected_city} ({selected_year})",
                 labels={"Vereinfachte_Straftat": "Straftat", "Number": "Anzahl"},
                 height=600)

    st.plotly_chart(fig)

def plot_gender_fraction(df_filtered, selected_city, selected_year):
    """Plots gender fraction for different crimes as a stacked bar chart."""
    
    if df_filtered.empty:
        st.warning(f"Keine Daten verfügbar für {selected_city} in {selected_year}.")
        return

    # Aggregate data if "All Cities" is selected
    if selected_city == "Alle Städte":
        df_filtered = df_filtered.groupby("Vereinfachte_Straftat", as_index=False)[["Tatverdaechtige_maennlich", "Tatverdaechtige_weiblich"]].sum()

    # Compute gender fraction
    df_filtered["Total"] = df_filtered["Tatverdaechtige_maennlich"] + df_filtered["Tatverdaechtige_weiblich"]
    df_filtered["Male Fraction"] = df_filtered["Tatverdaechtige_maennlich"] / df_filtered["Total"]
    df_filtered["Female Fraction"] = df_filtered["Tatverdaechtige_weiblich"] / df_filtered["Total"]

    # Sort by gender imbalance (largest difference first)
    df_filtered = df_filtered.sort_values(by="Male Fraction", ascending=False)
    # Convert to long format
    df_long = df_filtered.melt(id_vars=["Vereinfachte_Straftat"], 
                               value_vars=["Male Fraction", "Female Fraction"],
                               var_name="Gender", value_name="Fraction")

    # Rename gender labels
    df_long["Gender"] = df_long["Gender"].replace({
        "Male Fraction": "Männlich",
        "Female Fraction": "Weiblich"
    })

    # Create stacked bar chart
    fig = px.bar(df_long, 
                 x="Vereinfachte_Straftat", 
                 y="Fraction", 
                 color="Gender",
                 barmode="relative",  # Stacked bar chart
                 title=f"Geschlechtsverhältnis der Tatverdächtigen in {selected_city} ({selected_year})",
                 labels={"Vereinfachte_Straftat": "Straftat", "Fraction": "Geschlechtsverhältnis"},
                 height=600)

    st.plotly_chart(fig)