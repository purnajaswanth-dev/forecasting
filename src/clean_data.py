import pandas as pd


def clean_data(solar, weather, sites, monthly):
    """
    Clean all datasets before preprocessing.
    """

    # -----------------------------
    # Solar Dataset
    # -----------------------------
    solar = solar.copy()

    solar["Timestamp"] = pd.to_datetime(solar["Timestamp"])

    solar = solar.drop_duplicates()

    solar = solar.sort_values(
        by=["CampusKey", "SiteKey", "Timestamp"]
    )

    # -----------------------------
    # Weather Dataset
    # -----------------------------
    weather = weather.copy()

    weather["Timestamp"] = pd.to_datetime(weather["Timestamp"])

    weather = weather.drop_duplicates()

    weather = weather.sort_values(
        by=["CampusKey", "Timestamp"]
    )

    # -----------------------------
    # Site Dataset
    # -----------------------------
    sites = sites.copy()

    sites = sites.drop_duplicates()

    # -----------------------------
    # Monthly Dataset
    # -----------------------------
    monthly = monthly.copy()

    monthly = monthly.drop_duplicates()

    return solar, weather, sites, monthly