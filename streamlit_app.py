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

st.set_page_config(layout="wide")
# Load Data (Assumed already cleaned)
@st.cache_data
def load_data():
    df_cases = pd.read_csv(f"Faelle_combined_clean.csv")#, encoding='ISO-8859-1', sep=';')
    df_victims = pd.read_csv(f"Opfer_Tabellen/final_processed_data.csv")#, encoding='ISO-8859-1', sep=';')
    df_perps = pd.read_csv(f"Tatverdaechtige_combined_clean.csv")#, encoding='ISO-8859-1', sep=';')
    return df_cases, df_victims, df_perps

df_cases, df_victims, df_perps = load_data()

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
                            title="Crime Cases Map", mapbox_style="carto-positron", zoom=4.9, height=700, width=1000)
    
    fig.update_layout(clickmode='event+select')  # Enable click events
    return fig

def plot_crimes_vs_inhabitants():
    df_cases_filtered = df_cases[(df_cases["HZ"] > 0) & (df_cases['Vereinfachte_Straftat'] == "Straftaten insgesamt")]  # Avoid division by zero
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
    
    st.plotly_chart(fig, use_container_width=True)
    
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

    # Display the results
    st.write(f"Mean Squared Error (MSE) on validation data: {mse:.2f}")
    st.write(f"Regression Coefficients: Intercept = {model.intercept_:.2f}, Slope = {model.coef_[0]:.2f}")

    # Plot the regression line along with the data points
    fig_regression = px.scatter(df_cases_filtered, x="Inhabitants", y="Faelle", hover_name="Stadt",
                                title="Crimes vs. Inhabitants with Regression Line",
                                labels={"Inhabitants": "Number of Inhabitants", "Faelle": "Number of Crimes"})
    fig_regression.add_scatter(x=df_cases_filtered["Inhabitants"], 
                               y=model.predict(df_cases_filtered["Inhabitants"].values.reshape(-1, 1)), 
                               mode='lines', name="Regression Line", line=dict(color='red'))
    st.plotly_chart(fig_regression, use_container_width=True)


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

# Function to create the main overview page
def create_overview_page():
    st.title("👮‍♀️ krimistat")
    st.markdown("### Eine visuelle Erkundung der Kriminalitätsstatistik der Deutschen Polizei von 2016 bis 2023.")
    st.divider()


    years = sorted(df_cases['Jahr'].unique(), reverse=True)
    crime_types = df_cases['Vereinfachte_Straftat'].unique()

    st.sidebar.title("Options")
    selected_year = st.sidebar.selectbox("Select Year", df_cases["Jahr"].unique(), index=0)
    selected_crime = "Straftaten insgesamt" #st.selectbox("Select Crime Type", crime_types)

    # Merge coordinates
    city_coords = get_city_coordinates()
    df_filtered_cases = df_cases[(df_cases['Jahr'] == selected_year) & (df_cases['Vereinfachte_Straftat'] == selected_crime)]
    df_cases_with_coords = df_filtered_cases.merge(city_coords, on='Stadt', how='left')
    c1, c2 = st.columns(2)
    with c1:
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

        st.info("krimistat ist eine graphische Auswertung der Kriminalitätsstatistik der deutschen Polizei von 2016 bis 2023.") 
        st.info('📌 [Mehr zur Polizeilichen Kriminalstatistik beim BKA](https://www.bka.de/DE/AktuelleInformationen/StatistikenLagebilder/PolizeilicheKriminalstatistik/pks_node.html)', icon="ℹ️")

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
    st.title("🔪 Fälle")
    st.markdown("### Visualisierung der Falldaten")
    st.divider()

    years = sorted(df_cases['Jahr'].unique(), reverse=True)
    crime_types = df_cases['Straftat'].unique()
    
    selected_year = st.selectbox("Select Year", years, index=0)
    selected_crime = st.selectbox("Select Crime Type", crime_types)
    
    df_filtered_cases = df_cases[(df_cases['Jahr'] == selected_year) & (df_cases['Straftat'] == selected_crime)]
    
    st.metric("Total Cases", df_filtered_cases['Faelle'].sum())
    fig_cases = px.bar(df_filtered_cases, x='Stadt', y='Faelle', title="Cases per City")
    st.plotly_chart(fig_cases)
    st.dataframe(df_filtered_cases)
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
    fig_victims = px.pie(df_filtered_victims, values='Oper insgesamt', names='Fallstatus', title="Victim Distribution")
    st.plotly_chart(fig_victims)
    st.dataframe(df_filtered_victims)

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