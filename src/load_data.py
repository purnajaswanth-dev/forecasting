import pandas as pd
from config import RAW_DATA

def load_data():

    solar = pd.read_csv(RAW_DATA / "Solar_Energy_Generation.csv")

    weather = pd.read_csv(RAW_DATA / "Weather_Data_reordered_all.csv")

    sites = pd.read_csv(RAW_DATA / "Solar_Site_Details.csv")

    monthly = pd.read_csv(RAW_DATA / "Monthly_Summary_Solar.csv")

    return solar, weather, sites, monthly