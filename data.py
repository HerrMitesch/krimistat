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


# Load Data 
def load_data():
    df_cases = pd.read_csv(f"Faelle_combined_clean.csv")#, encoding='ISO-8859-1', sep=';')
    df_victims = pd.read_csv(f"Opfer_Tabell/Opfer_combined_clean_2016_2023.csv", encoding='ISO-8859-1', sep=';')
    df_perps = pd.read_csv(f"Tatverdaechtige_combined_clean.csv")#, encoding='ISO-8859-1', sep=';')
    return df_cases, df_victims, df_perps



def load_dashboard_data():
    return pd.read_csv("Opfer_combined_clean_2016_2023.csv", encoding='ISO-8859-1', sep=';')


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