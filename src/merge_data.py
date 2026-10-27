import pandas as pd

def merge_data(solar, weather, sites):
    # Merge solar with weather
    data = solar.merge(
        weather,
        on=["CampusKey", "Timestamp"],
        how="left"
    )

    # Merge site information
    data = data.merge(
        sites[["CampusKey", "SiteKey", "lat", "Lon"]],
        on=["CampusKey", "SiteKey"],
        how="left"
    )

    return data