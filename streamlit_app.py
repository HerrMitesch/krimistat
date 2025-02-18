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

st.set_page_config(layout="wide")
# Load Data (Assumed already cleaned)
@st.cache_data
def load_data():
    df_cases = pd.read_csv(f"Faelle_combined_clean.csv")#, encoding='ISO-8859-1', sep=';')
    df_victims = pd.read_csv(f"Opfer_Tabell/Opfer_combined_clean_2016_2023.csv", encoding='ISO-8859-1', sep=';')
    df_perps = pd.read_csv(f"Tatverdaechtige_combined_clean.csv")#, encoding='ISO-8859-1', sep=';')
    #return df_cases, df_victims
    return df_cases, df_victims, df_perps

df_cases, df_victims, df_perps = load_data()
#df_cases, df_victims= load_data()


#Import dashboard
@st.cache_data
def load_dashboard_data():
    return pd.read_csv("Opfer_combined_clean_2016_2023.csv", encoding='ISO-8859-1', sep=';')

df_dashboard = load_dashboard_data()

def load_germany_map():
    return gpd.read_file("https://raw.githubusercontent.com/johan/world.geo.json/master/countries/DEU.geo.json")

def get_city_coordinates():
    return pd.DataFrame([
    {'Stadt': 'Aachen', 'lat': 50.775555, 'lon': 6.083611},
    {'Stadt': 'Augsburg', 'lat': 48.366512, 'lon': 10.894446},
    {'Stadt': 'Bergisch Gladbach', 'lat': 51.099998, 'lon': 7.116667},
    {'Stadt': 'Berlin', 'lat': 52.520008, 'lon': 13.404954},
    {'Stadt': 'Bielefeld', 'lat': 52.021111, 'lon': 8.534722},
    {'Stadt': 'Bochum', 'lat': 51.481846, 'lon': 7.216236},
    {'Stadt': 'Bonn', 'lat': 50.733334, 'lon': 7.100000},
    {'Stadt': 'Bottrop', 'lat': 51.524723, 'lon': 6.922778},
    {'Stadt': 'Braunschweig', 'lat': 52.266666, 'lon': 10.516667},
    {'Stadt': 'Bremen', 'lat': 53.073635, 'lon': 8.806422},
    {'Stadt': 'Bremerhaven', 'lat': 53.549999, 'lon': 8.583333},
    {'Stadt': 'Chemnitz', 'lat': 50.833332, 'lon': 12.916667},
    {'Stadt': 'Darmstadt', 'lat': 49.878708, 'lon': 8.646927},
    {'Stadt': 'Dortmund', 'lat': 51.514244, 'lon': 7.468429},
    {'Stadt': 'Dresden', 'lat': 51.050407, 'lon': 13.737262},
    {'Stadt': 'Duisburg', 'lat': 51.435146, 'lon': 6.762692},
    {'Stadt': 'Düsseldorf', 'lat': 51.233334, 'lon': 6.783333},
    {'Stadt': 'Erfurt', 'lat': 50.983334, 'lon': 11.033333},
    {'Stadt': 'Erlangen', 'lat': 49.583332, 'lon': 11.016667},
    {'Stadt': 'Essen', 'lat': 51.450832, 'lon': 7.013056},
    {'Stadt': 'Frankfurt am Main', 'lat': 50.110924, 'lon': 8.682127},
    {'Stadt': 'Freiburg im Breisgau', 'lat': 47.997791, 'lon': 7.842609},
    {'Stadt': 'Fürth', 'lat': 49.466667, 'lon': 11.000000},
    {'Stadt': 'Gelsenkirchen', 'lat': 51.516666, 'lon': 7.100000},
    {'Stadt': 'Göttingen', 'lat': 51.545483, 'lon': 9.905548},
    {'Stadt': 'Hagen', 'lat': 51.366669, 'lon': 7.483333},
    {'Stadt': 'Halle (Saale)', 'lat': 51.483334, 'lon': 11.966667},
    {'Stadt': 'Hamburg', 'lat': 53.551086, 'lon': 9.993682},
    {'Stadt': 'Hamm', 'lat': 51.683334, 'lon': 7.816667},
    {'Stadt': 'Hannover', 'lat': 52.373920, 'lon': 9.735603},
    {'Stadt': 'Heidelberg', 'lat': 49.398750, 'lon': 8.672434},
    {'Stadt': 'Heilbronn', 'lat': 49.150002, 'lon': 9.216600},
    {'Stadt': 'Herne', 'lat': 51.549999, 'lon': 7.216667},
    {'Stadt': 'Ingolstadt', 'lat': 48.766666, 'lon': 11.433333},
    {'Stadt': 'Jena', 'lat': 50.927223, 'lon': 11.586111},
    {'Stadt': 'Karlsruhe', 'lat': 49.006889, 'lon': 8.403653},
    {'Stadt': 'Kassel', 'lat': 51.312801, 'lon': 9.481544},
    {'Stadt': 'Kiel', 'lat': 54.323334, 'lon': 10.139444},
    {'Stadt': 'Koblenz', 'lat': 50.360023, 'lon': 7.589907},
    {'Stadt': 'Köln', 'lat': 50.935173, 'lon': 6.953101},
    {'Stadt': 'Krefeld', 'lat': 51.333332, 'lon': 6.566667},
    {'Stadt': 'Leipzig', 'lat': 51.340199, 'lon': 12.360103},
    {'Stadt': 'Leverkusen', 'lat': 51.033333, 'lon': 6.983333},
    {'Stadt': 'Lübeck', 'lat': 53.869720, 'lon': 10.686389},
    {'Stadt': 'Ludwigshafen am Rhein', 'lat': 49.477409, 'lon': 8.445180},
    {'Stadt': 'Magdeburg', 'lat': 52.133331, 'lon': 11.616667},
    {'Stadt': 'Mainz', 'lat': 49.992863, 'lon': 8.247253},
    {'Stadt': 'Mannheim', 'lat': 49.488888, 'lon': 8.469167},
    {'Stadt': 'Moers', 'lat': 51.451603, 'lon': 6.640815},
    {'Stadt': 'Mönchengladbach', 'lat': 51.200001, 'lon': 6.433333},
    {'Stadt': 'Mülheim an der Ruhr', 'lat': 51.433334, 'lon': 6.883333},
    {'Stadt': 'München', 'lat': 48.137154, 'lon': 11.576124},
    {'Stadt': 'Münster', 'lat': 51.962510, 'lon': 7.625187},
    {'Stadt': 'Neuss', 'lat': 51.200001, 'lon': 6.683333},
    {'Stadt': 'Nürnberg', 'lat': 49.452103, 'lon': 11.076665},
    {'Stadt': 'Oberhausen', 'lat': 51.466667, 'lon': 6.850000},
    {'Stadt': 'Offenbach am Main', 'lat': 50.099998, 'lon': 8.766667},
    {'Stadt': 'Oldenburg (Oldenburg)', 'lat': 53.143890, 'lon': 8.213889},
    {'Stadt': 'Osnabrück', 'lat': 52.279911, 'lon': 8.047178},
    {'Stadt': 'Paderborn', 'lat': 51.719051, 'lon': 8.754384},
    {'Stadt': 'Pforzheim', 'lat': 48.884548, 'lon': 8.698479},
    {'Stadt': 'Potsdam', 'lat': 52.400002, 'lon': 13.066667},
    {'Stadt': 'Recklinghausen', 'lat': 51.614063, 'lon': 7.197946},
    {'Stadt': 'Regensburg', 'lat': 49.013432, 'lon': 12.101624},
    {'Stadt': 'Remscheid', 'lat': 51.178478, 'lon': 7.189698},
    {'Stadt': 'Reutlingen', 'lat': 48.49144, 'lon': 9.20427},
    {'Stadt': 'Rostock', 'lat': 54.092423, 'lon': 12.099147},
    {'Stadt': 'Saarbrücken', 'lat': 49.235401, 'lon': 6.981650},
    {'Stadt': 'Siegen', 'lat': 50.87481, 'lon': 8.02431},
    {'Stadt': 'Solingen', 'lat': 51.171677, 'lon': 7.083333},
    {'Stadt': 'Stuttgart', 'lat': 48.775845, 'lon': 9.182932},
    {'Stadt': 'Trier', 'lat': 49.750000, 'lon': 6.637143},
    {'Stadt': 'Ulm', 'lat': 48.398407, 'lon': 9.991550},
    {'Stadt': 'Wiesbaden', 'lat': 50.082580, 'lon': 8.249323},
    {'Stadt': 'Wolfsburg', 'lat': 52.422648, 'lon': 10.786546},
    {'Stadt': 'Wuppertal', 'lat': 51.256213, 'lon': 7.150764},
    {'Stadt': 'Würzburg', 'lat': 49.791304, 'lon': 9.953355},
    {'Stadt': 'Cottbus', 'lat': 51.757712, 'lon': 14.328880},
    {'Stadt': 'Gütersloh', 'lat': 51.906927, 'lon': 8.378686},
    {'Stadt': 'Kaiserslautern', 'lat': 49.440102, 'lon': 7.749126},
    {'Stadt': 'Hanau', 'lat': 50.132543, 'lon': 8.916687},
    {'Stadt': 'Rostock, Hanse- und Universitätsstadt', 'lat': 54.092423, 'lon': 12.099147},]
    )

def get_city_population(city):
    return random.randint(50000, 1000000)

# Placeholder function for city images
def get_city_image(city):
    return "https://via.placeholder.com/400"

# Function to create the map
def create_map(df_filtered_cases):
    fig = px.scatter_mapbox(df_filtered_cases, lat='lat', lon='lon', size='Faelle', hover_name='Stadt',
                            title="Straftaten Karte", mapbox_style="carto-positron", zoom=4.9, height=700, width=1000)
    
    fig.update_layout(clickmode='event+select')  # Enable click events
    return fig

def plot_crimes_vs_inhabitants():
    df_cases_filtered = df_cases[(df_cases["HZ"] > 0) & (df_cases['Straftat'] == "Straftaten insgesamt")]  # Avoid division by zero
    df_cases_filtered["Inhabitants"] = (df_cases_filtered["Faelle"] * 100000) / df_cases_filtered["HZ"]

    fig = px.scatter(df_cases_filtered, x="Inhabitants", y="Faelle", hover_name="Stadt",
                     title="Crimes vs. Inhabitants per City",
                     labels={"Inhabitants": "Number of Inhabitants", "Faelle": "Number of Crimes"})

    st.plotly_chart(fig, use_container_width=True)    

def plot_crimes_vs_inhabitants():
    # Filter out rows where "HZ" is 0 to avoid division by zero
    df_cases_filtered = df_cases[(df_cases["HZ"] > 0) & (df_cases["Straftat"]=="Straftaten insgesamt")]  
    df_cases_filtered["Inhabitants"] = (df_cases_filtered["Faelle"] * 100000) / df_cases_filtered["HZ"]

    # Create the scatter plot
    fig = px.scatter(df_cases_filtered, x="Inhabitants", y="Faelle", hover_name="Stadt",
                     title="Crimes vs. Inhabitants per City",
                     labels={"Inhabitants": "Number of Inhabitants", "Faelle": "Number of Crimes"})
    
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
                                title="Crimes vs. Inhabitants with Regression Line",
                                labels={"Inhabitants": "Number of Inhabitants", "Faelle": "Number of Crimes"})
        fig_regression.add_scatter(x=df_cases_filtered["Inhabitants"], 
                               y=model.predict(df_cases_filtered["Inhabitants"].values.reshape(-1, 1)), 
                               mode='lines', name="Regression Line", line=dict(color='red'))
        st.plotly_chart(fig_regression, use_container_width=True)
        # Display the results
        #st.write(f"Mean Squared Error (MSE) on validation data: {mse:.2f}")
        #st.write(f"Regression Coefficients: Intercept = {model.intercept_:.2f}, Slope = {model.coef_[0]:.2f}")
    else:
        st.plotly_chart(fig, use_container_width=True)
    
    
    
    


#def plot_crimes_vs_inhabitants():
    # Filter out rows where "HZ" is 0 to avoid division by zero
#    df_cases_filtered = df_cases[(df_cases["HZ"] > 0) & (df_cases["Straftat"]=="Straftaten insgesamt")]  
#    df_cases_filtered["Inhabitants"] = (df_cases_filtered["Faelle"] * 100000) / df_cases_filtered["HZ"]

    # Create the scatter plot
#    fig = px.scatter(df_cases_filtered, x="Inhabitants", y="Faelle", hover_name="Stadt",
#                     title="Crimes vs. Inhabitants per City",
#                     labels={"Inhabitants": "Number of Inhabitants", "Faelle": "Number of Crimes"})
    
#    st.plotly_chart(fig, use_container_width=True)
    
    # Prepare the data for training
#    X = df_cases_filtered["Inhabitants"].values.reshape(-1, 1)  # Features (Inhabitants)
#    y = df_cases_filtered["Faelle"].values  # Target variable (Number of Crimes)
    
    # Split data into 90% training and 10% validation
#    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initialize the Random Forest Regressor
#    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model on the training data
#    rf_model.fit(X_train, y_train)

    # Make predictions on the validation data
#    y_pred = rf_model.predict(X_val)

    # Calculate the Mean Squared Error (MSE) on the validation data
#    mse = mean_squared_error(y_val, y_pred)

    # Calculate the R-squared (R²) score for the validation data
#    r_squared = r2_score(y_val, y_pred)

    # Display the results
#    st.write(f"Mean Squared Error (MSE) on validation data: {mse:.2#f}")
#    st.write(f"R-squared (R²) on validation data: {r_squared:.2f}")

#    # Plot the predicted vs actual values
#    fig_rf = px.scatter(df_cases_filtered, x="Inhabitants", y="Faelle", hover_name="Stadt",
#                        title="Crimes vs. Inhabitants with Random Forest Predictions",
#                        labels={"Inhabitants": "Number of Inhabitants", "Faelle": "Number of Crimes"})
#    fig_rf.add_scatter(x=X_val.flatten(), y=y_pred, mode='markers', name="Predictions", marker=dict(color='red'))
#    st.plotly_chart(fig_rf, use_container_width=True) 

# Function to display city information
def display_city_info(city):
    st.subheader(f"Crime Statistics for {city}")
    inhabitants = get_city_population(city)
    st.write(f"Number of Inhabitants: {inhabitants}")

    city_image_url = get_city_image(city)
    st.image(city_image_url, caption=f"Characteristic view of {city}")

    city_crime_data = df_cases[df_cases['Stadt'] == city].groupby(['Jahr', 'Straftat'])['Faelle'].sum().reset_index()
    top_crimes = city_crime_data.groupby('Straftat')['Faelle'].sum().nlargest(10).index
    city_crime_data = city_crime_data[city_crime_data['Straftat'].isin(top_crimes)]

    fig_crime_trends = px.line(city_crime_data, x='Jahr', y='Faelle', color='Straftat', title=f"Top 10 Crimes in {city} Over Time")
    st.plotly_chart(fig_crime_trends)

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
        "Tatverdaechtige_maennlich": "Male",
        "Tatverdaechtige_weiblich": "Female"
    })

    # Create grouped bar chart
    fig = px.bar(df_long, 
                 x="Vereinfachte_Straftat", 
                 y="Number", 
                 color="Gender",
                 barmode="group", 
                 title=f"Gender Distribution of Crimes in {selected_city} ({selected_year})",
                 labels={"Vereinfachte_Straftat": "Crime Type", "Number": "Number of Suspects"},
                 height=600)

    st.plotly_chart(fig)

def plot_gender_fraction(df_filtered, selected_city, selected_year):
    """Plots gender fraction for different crimes as a stacked bar chart."""
    
    if df_filtered.empty:
        st.warning(f"No data available for {selected_city} in {selected_year}.")
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
        "Male Fraction": "Male",
        "Female Fraction": "Female"
    })

    # Create stacked bar chart
    fig = px.bar(df_long, 
                 x="Vereinfachte_Straftat", 
                 y="Fraction", 
                 color="Gender",
                 barmode="relative",  # Stacked bar chart
                 title=f"Gender Fraction of Crimes in {selected_city} ({selected_year})",
                 labels={"Vereinfachte_Straftat": "Crime Type", "Fraction": "Gender Proportion"},
                 height=600)

    st.plotly_chart(fig)


# Function to create the main overview page
#def create_overview_page():
    #st.title("👮‍♀️ krimistat")
    #st.markdown("### Eine visuelle Erkundung der Kriminalitätsstatistik der Deutschen Polizei von 2016 bis 2023.")
    #st.divider()



def create_overview_page():
    st.title("Crime Data Exploration - German BKA")
    st.image(r"C:\Users\PC\Desktop\Projekt_Weiterbildung\krimistat\Opfer_Tabell\invest.webp", use_container_width=True)
    st.subheader("Summary Metrics")
    
    total_cases = df_cases['Faelle'].sum() if 'Faelle' in df_cases.columns else "N/A"
    total_victims = df_victims['Oper insgesamt'].sum() if 'Oper insgesamt' in df_victims.columns else "N/A"
    total_perpetratrors = df_perps['gesamt'].sum() if 'gesamt' in df_perps.columns else "N/A"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cases", total_cases)
    with col2:
        st.metric("Total Victims", total_victims)
    with col3:
        st.metric("Total Perpetrators", total_perpetratrors)
    
    st.write(
        """
        Willkommen bei der Crime Data Exploration. Nutzen Sie die Seitenleiste, um zwischen den verschiedenen Seiten zu navigieren. 
        Entdecken Sie die Gesamtdaten auf der Übersichtsseite, detaillierte Fallinformationen auf der Fälle-Seite, 
        Opferdetails auf der Opfer-Seite und schließlich Täterdetails auf der Täter-Seite."
        """
    )
     
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

    st.sidebar.title("Options")
    selected_year = st.sidebar.slider("Select Year", min_value=2016, max_value = 2023, value = 2020)

    c1, c2 = st.columns(2)
    with c1:
        
        selected_crime = st.selectbox("Select Crime Type", crime_types)
        city_coords = get_city_coordinates()
        df_filtered_cases = df_cases[(df_cases['Jahr'] == selected_year) & (df_cases['Vereinfachte_Straftat'] == selected_crime)]
        df_cases_with_coords = df_filtered_cases.merge(city_coords, on='Stadt', how='left')
        fig_map = create_map(df_cases_with_coords)
        click_data = st.plotly_chart(fig_map, use_container_width=True)

    with c2: 
        # Filter by selected year and remove rows where "Vereinfachte_Straftat" contains "insgesamt"
        filtered_df = df_cases[(df_cases['Jahr'] == selected_year) & ~df_cases['Vereinfachte_Straftat'].str.contains("insgesamt", case=False, na=False)]

        # Aggregate by summing the "Faelle" column
        top_crimes = filtered_df.groupby("Vereinfachte_Straftat")["Faelle"].sum().nlargest(10)

        # Plot if there is data
        if not top_crimes.empty:
            fig_pie = px.pie(top_crimes, values=top_crimes.values, names=top_crimes.index, title="Top 10 Most Common Crimes")
            st.plotly_chart(fig_pie)
        else:
            st.warning(f"No crime data available for {selected_year}.")   
    st.divider()
    # City filter
    cities = ["Alle Städte"] + sorted(df_cases["Stadt"].unique().tolist())
    selected_city = st.selectbox("Select a city:", cities, index=0)
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
    plot_crimes_vs_inhabitants()

def create_victims_page():
    st.title("😢 Opfer")
    st.markdown("### Visualisierung der Opferdaten")
    st.divider()
    years = sorted(df_victims['Jahr'].unique(), reverse=True)
    crime_types = df_victims['Straftat'].unique()
    
    selected_year = st.selectbox("Select Year", years, index=0)
    selected_crime = st.selectbox("Select Crime Type", crime_types)
    
    df_filtered_victims = df_victims[(df_victims['Jahr'] == selected_year) & (df_victims['Straftat'] == selected_crime)]
    
    st.metric("Total Victims", df_filtered_victims['Oper insgesamt'].sum())
    
    # Load Data (Assuming df_dashboard is already loaded)
    st.title("Crime Dashboard")

    # City Filter
    city_list = ["All"] + sorted(df_dashboard["Stadt"].unique().tolist())
    selected_city = st.selectbox("Select a City", city_list)

    # Filter Data
    if selected_city != "All":
        
        df_filtered = df_dashboard[df_dashboard["Stadt"] == selected_city]
    else:
        df_filtered = df_dashboard

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Crime Trends Over the Years
    crime_trend = df_filtered.groupby("Jahr")["Oper insgesamt"].sum()
    sns.lineplot(x=crime_trend.index, y=crime_trend.values, marker="o", ax=axes[0, 0])
    axes[0, 0].set_title("Crime Trends (2014-2023)")
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Total Crimes")

    # Male vs Female Victim Trends
    crime_trend_male = df_filtered.groupby("Jahr")["Opfer maennlich"].sum()
    crime_trend_female = df_filtered.groupby("Jahr")["Opfer weiblich"].sum()
    sns.lineplot(x=crime_trend_male.index, y=crime_trend_male.values, marker="o", label="Male Victims", ax=axes[0, 1])
    sns.lineplot(x=crime_trend_female.index, y=crime_trend_female.values, marker="s", label="Female Victims", ax=axes[0, 1])
    axes[0, 1].set_title("Male vs Female Victim Trends (2014-2023)")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Total Victims")
    axes[0, 1].legend()

    # Crime Heatmap (Top 20 Cities)
    top_20_cities = df_dashboard.groupby("Stadt")["Oper insgesamt"].sum().nlargest(20).index
    df_filtered_top20 = df_dashboard[df_dashboard["Stadt"].isin(top_20_cities)]
    df_pivot = df_filtered_top20.pivot_table(values="Oper insgesamt", index="Stadt", columns="Jahr", aggfunc="sum", fill_value=0)
    sns.heatmap(df_pivot, cmap="Blues", linewidths=0.5, ax=axes[1, 0])
    axes[1, 0].set_title("Victims by City Over Time")

    # Victim Distribution by Age Category
    age_categories = {
    "Opfer - Kinder bis unter 6 Jahre - insgesamt": "0-5 years",
    "Opfer Kinder 6 bis unter 14 Jahre - insgesamt": "6-13 years",
    "Opfer Jugendliche 14 bis unter 18 Jahre - insgesamt": "14-17 years",
    "Opfer - Heranwachsende 18 bis unter 21 Jahre - insgesamt": "18-20 years",
    "Opfer Erwachsene 21 bis unter 60 Jahre - insgesamt": "21-59 years",
    "Opfer - Erwachsene 60 Jahre und aelter - insgesamt": "60+ years"
}
    df_age_victims = df_filtered[list(age_categories.keys())].sum().rename(index=age_categories)
    sns.barplot(x=df_age_victims.index, y=df_age_victims.values, palette="Blues", ax=axes[1, 1])
    axes[1, 1].set_title("Victim Distribution by Age Category")
    axes[1, 1].set_xlabel("Age Category")
    axes[1, 1].set_ylabel("Number of Victims")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

    # Display plots
    st.pyplot(fig)
    
    

    
        
    

     
    
    
    
    
    #fig_victims = px.pie(df_filtered_victims, values='Oper insgesamt', names='Fallstatus', title="Victim Distribution")
    #st.plotly_chart(fig_victims)
    #st.dataframe(df_filtered_victims)


def create_perpetrators_page():
    st.title("🦹‍♂️ Tatverdächtige")
    st.markdown("### Visualisierung der Daten zu Tatverdächtigen")
    st.divider()

    years = sorted(df_perps['Jahr'].unique(), reverse=True)
    crime_types = df_perps['Straftat'].unique()

    
    selected_year = st.selectbox("Select Year", years, index=0)
    selected_crime = st.selectbox("Select Crime Type", crime_types)
    
    df_filtered_perps = df_perps[(df_perps['Jahr'] == selected_year) & (df_perps['Straftat'] == selected_crime)]
    
    st.metric("Total Perpetrators", df_filtered_perps['gesamt'].sum())
    fig_perps = px.histogram(df_filtered_perps, x='Erwachsene_gesamt', title="Age Distribution of Perpetrators")
    st.plotly_chart(fig_perps)
    st.dataframe(df_filtered_perps)

# Sidebar Navigation
st.sidebar.title("Navigation")


page = st.sidebar.radio("Go to", ["Übersicht", "Rohdaten", "Fälle", "Opfer", "Täter"])


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