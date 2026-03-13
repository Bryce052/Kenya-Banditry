# ============================================================
# KENYA LEAFLET MAPS FOR INCIDENT CONCENTRATION
# ============================================================
# PURPOSE
# Build interactive Leaflet maps in Python (via folium) for a Kenya
# incident dataset, showing event concentration and relevant variables.
#
# MAPS CREATED
# 1. Event point map
# 2. Event cluster map
# 3. Heatmap of event concentration
# 4. Severity-weighted heatmap
# 5. Death-weighted heatmap
# 6. Injury-weighted heatmap
# 7. Cattle stolen heatmap
# 8. Abductions heatmap
# 9. Displacement heatmap
# 10. Weapon-type layered map
# 11. Year-filtered maps
# 12. Choropleth-style county concentration map (if county extracted)
#
# OUTPUT
# All maps are saved as HTML files in:
#   kenya_leaflet_outputs/
#
# REQUIRED PACKAGES
# pip install pandas numpy folium geopy branca
#
# OPTIONAL BUT USEFUL
# pip install geopandas shapely
#
# IMPORTANT
# - Replace FILE_NAME with your CSV filename
# - If your dataset does not contain latitude/longitude, this script
#   will try to geocode place_name against Kenya using geopy.
# - Geocoding can be slow and imperfect.
#
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import folium

from folium.plugins import HeatMap, MarkerCluster
from branca.colormap import linear

warnings.filterwarnings("ignore")

try:
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except Exception:
    GEOPY_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================
FILE_NAME = "kimanalysis.csv"   # <-- CHANGE THIS
OUTPUT_DIR = "kenya_leaflet_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KENYA_CENTER = [0.0236, 37.9062]
DEFAULT_ZOOM = 6

# If geocoding is needed, pause between requests
GEOCODE_SLEEP_SECONDS = 1.1

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(FILE_NAME)

print("Loaded file:", FILE_NAME)
print("Shape:", df.shape)

# ============================================================
# 2. STANDARDIZE COLUMNS BY POSITION
#    Based on the exact variable order you supplied
# ============================================================
expected_columns = [
    "doc_code",
    "year",
    "month",
    "place_name",
    "num_attacking_bandits",
    "cattle_stolen",
    "total_deaths",
    "police_deaths",
    "local_deaths",
    "suspected_bandits_killed",
    "total_injuries",
    "police_injured",
    "locals_injured",
    "bandits_injured",
    "people_displaced",
    "people_abducted",
    "other_crime_1",
    "other_crime_2",
    "other_crime_3",
    "weapon_type",
    "weapon_details",
    "num_weapons",
    "firearm_ammunition_present",
    "ammunition_type",
    "facility_attacked",
    "weblink",
    "media_house",
    "comments"
]

rename_map = {}
for i, old_col in enumerate(df.columns[:len(expected_columns)]):
    rename_map[old_col] = expected_columns[i]
df = df.rename(columns=rename_map)

# ============================================================
# 3. CLEANING
# ============================================================
def clean_text(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "":
        return np.nan
    if x.lower() in ["unknown", "unk", "na", "n/a", "none", "nil", "nan"]:
        return np.nan
    return x

for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].apply(clean_text)

numeric_cols = [
    "year",
    "num_attacking_bandits",
    "cattle_stolen",
    "total_deaths",
    "police_deaths",
    "local_deaths",
    "suspected_bandits_killed",
    "total_injuries",
    "police_injured",
    "locals_injured",
    "bandits_injured",
    "people_displaced",
    "people_abducted",
    "num_weapons"
]

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

for c in ["weapon_type", "weapon_details", "facility_attacked", "media_house",
          "other_crime_1", "other_crime_2", "other_crime_3", "ammunition_type"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.upper().str.strip()
        df[c] = df[c].replace({"NAN": np.nan})

if "doc_code" not in df.columns:
    df["doc_code"] = [f"DOC_{i+1:05d}" for i in range(len(df))]

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================
def split_semicolon_values(value):
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).split(";") if v.strip()]

def safe_num(x):
    return 0 if pd.isna(x) else float(x)

def infer_county(place_text):
    """
    Very simple county extraction from semicolon-separated place field.
    Example:
      'TURKANA;EAST POKOT' -> 'TURKANA'
    """
    if pd.isna(place_text):
        return np.nan
    first = str(place_text).split(";")[0].strip().upper()
    return first if first else np.nan

def build_popup(row):
    other_crimes = "; ".join(
        [str(v) for v in [row.get("other_crime_1"), row.get("other_crime_2"), row.get("other_crime_3")] if pd.notna(v)]
    )
    popup_html = f"""
    <div style="width: 320px;">
        <b>Document Code:</b> {row.get('doc_code', '')}<br>
        <b>Year:</b> {row.get('year', '')}<br>
        <b>Month:</b> {row.get('month', '')}<br>
        <b>Place:</b> {row.get('place_name', '')}<br>
        <b>County (inferred):</b> {row.get('county_inferred', '')}<br>
        <b>Deaths:</b> {row.get('total_deaths', '')}<br>
        <b>Injuries:</b> {row.get('total_injuries', '')}<br>
        <b>Cattle Stolen:</b> {row.get('cattle_stolen', '')}<br>
        <b>Abducted:</b> {row.get('people_abducted', '')}<br>
        <b>Displaced:</b> {row.get('people_displaced', '')}<br>
        <b>Weapon Type:</b> {row.get('weapon_type', '')}<br>
        <b>Weapon Details:</b> {row.get('weapon_details', '')}<br>
        <b>Other Crime(s):</b> {other_crimes}<br>
        <b>Media House:</b> {row.get('media_house', '')}<br>
        <b>Weblink:</b> {row.get('weblink', '')}<br>
    </div>
    """
    return popup_html

def save_map(fmap, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fmap.save(path)
    print("Saved:", path)

def make_base_map(title=None):
    fmap = folium.Map(
        location=KENYA_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles=None,
        control_scale=True
    )

    # Leaflet base layers
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True
    ).add_to(fmap)

    folium.TileLayer(
        tiles="CartoDB positron",
        name="CartoDB Positron",
        control=True
    ).add_to(fmap)

    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="CartoDB Dark Matter",
        control=True
    ).add_to(fmap)

    if title:
        title_html = f"""
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
        """
        fmap.get_root().html.add_child(folium.Element(title_html))

    return fmap

def radius_from_value(value, min_r=4, max_r=16, divisor=5):
    if pd.isna(value):
        return min_r
    r = min_r + (float(value) / divisor)
    return max(min_r, min(max_r, r))

def color_from_severity(sev):
    if sev >= 50:
        return "darkred"
    if sev >= 25:
        return "red"
    if sev >= 10:
        return "orange"
    if sev >= 5:
        return "gold"
    return "blue"

# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
df["county_inferred"] = df["place_name"].apply(infer_county)

df["severity_index"] = (
    3 * df.get("total_deaths", pd.Series(0, index=df.index)).fillna(0)
    + 2 * df.get("total_injuries", pd.Series(0, index=df.index)).fillna(0)
    + 0.02 * df.get("cattle_stolen", pd.Series(0, index=df.index)).fillna(0)
    + 0.5 * df.get("people_abducted", pd.Series(0, index=df.index)).fillna(0)
    + 0.01 * df.get("people_displaced", pd.Series(0, index=df.index)).fillna(0)
    + 0.2 * df.get("num_attacking_bandits", pd.Series(0, index=df.index)).fillna(0)
)

df["event_weight"] = 1.0
df["death_weight"] = df.get("total_deaths", pd.Series(0, index=df.index)).fillna(0)
df["injury_weight"] = df.get("total_injuries", pd.Series(0, index=df.index)).fillna(0)
df["cattle_weight"] = df.get("cattle_stolen", pd.Series(0, index=df.index)).fillna(0)
df["abduction_weight"] = df.get("people_abducted", pd.Series(0, index=df.index)).fillna(0)
df["displacement_weight"] = df.get("people_displaced", pd.Series(0, index=df.index)).fillna(0)

# ============================================================
# 6. DETECT OR CREATE COORDINATES
# ============================================================
lat_candidates = [c for c in df.columns if c.lower() in ["latitude", "lat", "y"]]
lon_candidates = [c for c in df.columns if c.lower() in ["longitude", "lon", "lng", "long", "x"]]

if len(lat_candidates) > 0 and len(lon_candidates) > 0:
    lat_col = lat_candidates[0]
    lon_col = lon_candidates[0]
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    print("Using existing coordinates from columns:", lat_col, "and", lon_col)
else:
    df["latitude"] = np.nan
    df["longitude"] = np.nan

# ============================================================
# 7. GEOCODE PLACE NAMES IF NEEDED
# ============================================================
if df["latitude"].isna().all() or df["longitude"].isna().all():
    if GEOPY_AVAILABLE:
        print("No usable coordinate columns found. Attempting geocoding from place_name...")
        geolocator = Nominatim(user_agent="kenya_incident_leaflet_mapper")

        unique_places = df["place_name"].dropna().astype(str).unique().tolist()
        geocode_cache = {}

        for place in unique_places:
            try:
                query = f"{place}, Kenya"
                location = geolocator.geocode(query, timeout=15)
                if location is not None:
                    geocode_cache[place] = (location.latitude, location.longitude)
                else:
                    geocode_cache[place] = (np.nan, np.nan)
            except Exception:
                geocode_cache[place] = (np.nan, np.nan)

            time.sleep(GEOCODE_SLEEP_SECONDS)

        df["latitude"] = df["place_name"].map(lambda x: geocode_cache.get(x, (np.nan, np.nan))[0] if pd.notna(x) else np.nan)
        df["longitude"] = df["place_name"].map(lambda x: geocode_cache.get(x, (np.nan, np.nan))[1] if pd.notna(x) else np.nan)
    else:
        print("geopy is not installed, so geocoding cannot run.")

# ============================================================
# 8. FILTER TO ROWS WITH COORDINATES
# ============================================================
geo_df = df[df["latitude"].notna() & df["longitude"].notna()].copy()

# Restrict to approximate Kenya bbox to avoid bad geocodes
geo_df = geo_df[
    (geo_df["latitude"].between(-5.5, 5.5)) &
    (geo_df["longitude"].between(33.0, 42.5))
].copy()

print("Rows with usable Kenya coordinates:", len(geo_df))

geo_df.to_csv(os.path.join(OUTPUT_DIR, "geocoded_incidents.csv"), index=False)

# ============================================================
# 9. POINT MAP OF EVENTS
# ============================================================
if len(geo_df) > 0:
    fmap = make_base_map("Kenya Incident Event Points")

    for _, row in geo_df.iterrows():
        sev = safe_num(row["severity_index"])
        radius = radius_from_value(sev, min_r=4, max_r=14, divisor=8)

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            popup=folium.Popup(build_popup(row), max_width=400),
            tooltip=f"{row.get('doc_code', '')} | {row.get('place_name', '')}",
            color=color_from_severity(sev),
            weight=1,
            fill=True,
            fill_opacity=0.6
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    save_map(fmap, "01_event_points_map.html")

# ============================================================
# 10. CLUSTER MAP OF EVENTS
# ============================================================
if len(geo_df) > 0:
    fmap = make_base_map("Kenya Incident Cluster Map")
    cluster = MarkerCluster(name="Incident Clusters").add_to(fmap)

    for _, row in geo_df.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(build_popup(row), max_width=400),
            tooltip=f"{row.get('doc_code', '')} | {row.get('place_name', '')}"
        ).add_to(cluster)

    folium.LayerControl().add_to(fmap)
    save_map(fmap, "02_event_cluster_map.html")

# ============================================================
# 11. GENERIC HEATMAP FUNCTION
# ============================================================
def create_heatmap(dataframe, weight_col, title, filename, radius=15, blur=12, min_opacity=0.35):
    if len(dataframe) == 0:
        return

    fmap = make_base_map(title)

    heat_data = dataframe[["latitude", "longitude", weight_col]].fillna(0).values.tolist()
    HeatMap(
        heat_data,
        name=title,
        radius=radius,
        blur=blur,
        min_opacity=min_opacity,
        max_zoom=10
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    save_map(fmap, filename)

# ============================================================
# 12. EVENT CONCENTRATION MAPS
# ============================================================
if len(geo_df) > 0:
    create_heatmap(
        geo_df,
        weight_col="event_weight",
        title="Kenya Event Concentration Heatmap",
        filename="03_event_concentration_heatmap.html",
        radius=16,
        blur=12
    )

    create_heatmap(
        geo_df,
        weight_col="severity_index",
        title="Kenya Severity-Weighted Concentration Heatmap",
        filename="04_severity_heatmap.html",
        radius=18,
        blur=14
    )

    create_heatmap(
        geo_df,
        weight_col="death_weight",
        title="Kenya Death-Weighted Heatmap",
        filename="05_death_heatmap.html",
        radius=18,
        blur=14
    )

    create_heatmap(
        geo_df,
        weight_col="injury_weight",
        title="Kenya Injury-Weighted Heatmap",
        filename="06_injury_heatmap.html",
        radius=18,
        blur=14
    )

    create_heatmap(
        geo_df,
        weight_col="cattle_weight",
        title="Kenya Cattle Stolen Heatmap",
        filename="07_cattle_stolen_heatmap.html",
        radius=18,
        blur=14
    )

    create_heatmap(
        geo_df,
        weight_col="abduction_weight",
        title="Kenya Abduction Heatmap",
        filename="08_abduction_heatmap.html",
        radius=18,
        blur=14
    )

    create_heatmap(
        geo_df,
        weight_col="displacement_weight",
        title="Kenya Displacement Heatmap",
        filename="09_displacement_heatmap.html",
        radius=18,
        blur=14
    )

# ============================================================
# 13. BUBBLE MAPS FOR RELEVANT VARIABLES
# ============================================================
def create_bubble_map(dataframe, value_col, title, filename, fill_color="red"):
    if len(dataframe) == 0:
        return

    fmap = make_base_map(title)

    for _, row in dataframe.iterrows():
        value = safe_num(row.get(value_col))
        if value <= 0:
            continue

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius_from_value(value, min_r=4, max_r=18, divisor=max(1, np.nanmax([1, dataframe[value_col].fillna(0).median()]))),
            popup=folium.Popup(build_popup(row), max_width=400),
            tooltip=f"{row.get('place_name', '')} | {value_col}: {value}",
            color=fill_color,
            weight=1,
            fill=True,
            fill_opacity=0.55
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    save_map(fmap, filename)

if len(geo_df) > 0:
    create_bubble_map(geo_df, "total_deaths", "Kenya Bubble Map: Deaths", "10_bubble_map_deaths.html", fill_color="darkred")
    create_bubble_map(geo_df, "total_injuries", "Kenya Bubble Map: Injuries", "11_bubble_map_injuries.html", fill_color="orange")
    create_bubble_map(geo_df, "cattle_stolen", "Kenya Bubble Map: Cattle Stolen", "12_bubble_map_cattle.html", fill_color="green")
    create_bubble_map(geo_df, "people_abducted", "Kenya Bubble Map: Abductions", "13_bubble_map_abductions.html", fill_color="purple")
    create_bubble_map(geo_df, "people_displaced", "Kenya Bubble Map: Displacement", "14_bubble_map_displacement.html", fill_color="blue")
    create_bubble_map(geo_df, "severity_index", "Kenya Bubble Map: Severity Index", "15_bubble_map_severity.html", fill_color="black")

# ============================================================
# 14. WEAPON-TYPE LAYERED MAP
# ============================================================
if len(geo_df) > 0 and "weapon_type" in geo_df.columns:
    fmap = make_base_map("Kenya Weapon Type Layered Map")
    top_weapon_types = geo_df["weapon_type"].dropna().value_counts().head(8).index.tolist()

    weapon_colors = {
        "FIREARMS": "red",
        "BOW & ARROW": "green",
        "MACHETE": "purple"
    }

    for weapon in top_weapon_types:
        fg = folium.FeatureGroup(name=f"Weapon: {weapon}", show=True)
        subset = geo_df[geo_df["weapon_type"] == weapon]

        for _, row in subset.iterrows():
            color = weapon_colors.get(weapon, "blue")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius_from_value(row.get("severity_index"), min_r=4, max_r=14, divisor=8),
                popup=folium.Popup(build_popup(row), max_width=400),
                tooltip=f"{weapon} | {row.get('place_name', '')}",
                color=color,
                weight=1,
                fill=True,
                fill_opacity=0.6
            ).add_to(fg)

        fg.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    save_map(fmap, "16_weapon_type_layered_map.html")

# ============================================================
# 15. YEAR-SPECIFIC MAPS
# ============================================================
if len(geo_df) > 0 and "year" in geo_df.columns:
    years = sorted([int(y) for y in geo_df["year"].dropna().unique() if pd.notna(y)])

    for year in years:
        sub = geo_df[geo_df["year"] == year].copy()
        if len(sub) == 0:
            continue

        fmap = make_base_map(f"Kenya Event Concentration Heatmap - {year}")

        heat_data = sub[["latitude", "longitude", "event_weight"]].fillna(0).values.tolist()
        HeatMap(
            heat_data,
            name=f"Events {year}",
            radius=16,
            blur=12,
            min_opacity=0.35,
            max_zoom=10
        ).add_to(fmap)

        folium.LayerControl().add_to(fmap)
        save_map(fmap, f"17_heatmap_year_{year}.html")

# ============================================================
# 16. COUNTY CONCENTRATION MAP
#     This is a point-based county summary, not a boundary shapefile
#     choropleth. It still shows spatial concentration by inferred county.
# ============================================================
if len(geo_df) > 0 and "county_inferred" in geo_df.columns:
    county_summary = geo_df.groupby("county_inferred").agg(
        incidents=("doc_code", "count"),
        total_deaths=("total_deaths", "sum"),
        total_injuries=("total_injuries", "sum"),
        total_cattle_stolen=("cattle_stolen", "sum"),
        total_abducted=("people_abducted", "sum"),
        total_displaced=("people_displaced", "sum"),
        mean_latitude=("latitude", "mean"),
        mean_longitude=("longitude", "mean"),
        mean_severity=("severity_index", "mean")
    ).reset_index()

    county_summary = county_summary.dropna(subset=["mean_latitude", "mean_longitude"]).copy()
    county_summary.to_csv(os.path.join(OUTPUT_DIR, "county_concentration_summary.csv"), index=False)

    if len(county_summary) > 0:
        fmap = make_base_map("Kenya County Concentration Summary")

        max_incidents = county_summary["incidents"].max() if county_summary["incidents"].notna().any() else 1
        cmap = linear.YlOrRd_09.scale(1, max_incidents)
        cmap.caption = "Incident Count by Inferred County"

        for _, row in county_summary.iterrows():
            popup = f"""
            <div style="width:260px">
                <b>County:</b> {row['county_inferred']}<br>
                <b>Incidents:</b> {int(row['incidents'])}<br>
                <b>Total Deaths:</b> {safe_num(row['total_deaths'])}<br>
                <b>Total Injuries:</b> {safe_num(row['total_injuries'])}<br>
                <b>Total Cattle Stolen:</b> {safe_num(row['total_cattle_stolen'])}<br>
                <b>Total Abducted:</b> {safe_num(row['total_abducted'])}<br>
                <b>Total Displaced:</b> {safe_num(row['total_displaced'])}<br>
                <b>Mean Severity:</b> {round(safe_num(row['mean_severity']), 2)}<br>
            </div>
            """

            folium.CircleMarker(
                location=[row["mean_latitude"], row["mean_longitude"]],
                radius=radius_from_value(row["incidents"], min_r=6, max_r=20, divisor=2),
                popup=folium.Popup(popup, max_width=320),
                tooltip=f"{row['county_inferred']} | Incidents: {int(row['incidents'])}",
                color=cmap(row["incidents"]),
                weight=1,
                fill=True,
                fill_color=cmap(row["incidents"]),
                fill_opacity=0.75
            ).add_to(fmap)

        cmap.add_to(fmap)
        folium.LayerControl().add_to(fmap)
        save_map(fmap, "18_county_concentration_map.html")

# ============================================================
# 17. SINGLE MASTER MAP WITH LAYERS
# ============================================================
if len(geo_df) > 0:
    fmap = make_base_map("Kenya Incident Master Leaflet Map")

    # Event points layer
    fg_points = folium.FeatureGroup(name="Event Points", show=True)
    for _, row in geo_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius_from_value(row.get("severity_index"), min_r=4, max_r=14, divisor=8),
            popup=folium.Popup(build_popup(row), max_width=400),
            tooltip=f"{row.get('doc_code', '')} | {row.get('place_name', '')}",
            color=color_from_severity(safe_num(row.get("severity_index"))),
            weight=1,
            fill=True,
            fill_opacity=0.6
        ).add_to(fg_points)
    fg_points.add_to(fmap)

    # Event concentration layer
    fg_heat_events = folium.FeatureGroup(name="Heatmap: Event Concentration", show=False)
    HeatMap(
        geo_df[["latitude", "longitude", "event_weight"]].fillna(0).values.tolist(),
        radius=16,
        blur=12,
        min_opacity=0.35,
        max_zoom=10
    ).add_to(fg_heat_events)
    fg_heat_events.add_to(fmap)

    # Severity layer
    fg_heat_severity = folium.FeatureGroup(name="Heatmap: Severity", show=False)
    HeatMap(
        geo_df[["latitude", "longitude", "severity_index"]].fillna(0).values.tolist(),
        radius=18,
        blur=14,
        min_opacity=0.35,
        max_zoom=10
    ).add_to(fg_heat_severity)
    fg_heat_severity.add_to(fmap)

    # Deaths layer
    fg_heat_deaths = folium.FeatureGroup(name="Heatmap: Deaths", show=False)
    HeatMap(
        geo_df[["latitude", "longitude", "death_weight"]].fillna(0).values.tolist(),
        radius=18,
        blur=14,
        min_opacity=0.35,
        max_zoom=10
    ).add_to(fg_heat_deaths)
    fg_heat_deaths.add_to(fmap)

    # Injuries layer
    fg_heat_injuries = folium.FeatureGroup(name="Heatmap: Injuries", show=False)
    HeatMap(
        geo_df[["latitude", "longitude", "injury_weight"]].fillna(0).values.tolist(),
        radius=18,
        blur=14,
        min_opacity=0.35,
        max_zoom=10
    ).add_to(fg_heat_injuries)
    fg_heat_injuries.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    save_map(fmap, "19_master_leaflet_map.html")

# ============================================================
# 18. EXPORT SUMMARIES
# ============================================================
summary_rows = [
    ["rows_total", len(df)],
    ["rows_with_coordinates", len(geo_df)],
    ["unique_doc_codes", df["doc_code"].nunique(dropna=True)],
    ["unique_places", df["place_name"].nunique(dropna=True) if "place_name" in df.columns else np.nan],
    ["unique_inferred_counties", df["county_inferred"].nunique(dropna=True)],
    ["min_year", df["year"].min() if "year" in df.columns else np.nan],
    ["max_year", df["year"].max() if "year" in df.columns else np.nan],
    ["sum_total_deaths", df["total_deaths"].sum() if "total_deaths" in df.columns else np.nan],
    ["sum_total_injuries", df["total_injuries"].sum() if "total_injuries" in df.columns else np.nan],
    ["sum_cattle_stolen", df["cattle_stolen"].sum() if "cattle_stolen" in df.columns else np.nan],
    ["sum_people_abducted", df["people_abducted"].sum() if "people_abducted" in df.columns else np.nan],
    ["sum_people_displaced", df["people_displaced"].sum() if "people_displaced" in df.columns else np.nan]
]

summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
summary_df.to_csv(os.path.join(OUTPUT_DIR, "map_output_summary.csv"), index=False)

# top concentration tables
if "county_inferred" in geo_df.columns:
    geo_df["county_inferred"].value_counts().reset_index().rename(
        columns={"index": "county_inferred", "county_inferred": "incident_count"}
    ).to_csv(os.path.join(OUTPUT_DIR, "top_counties_by_incident_count.csv"), index=False)

if "place_name" in geo_df.columns:
    geo_df["place_name"].value_counts().reset_index().rename(
        columns={"index": "place_name", "place_name": "incident_count"}
    ).to_csv(os.path.join(OUTPUT_DIR, "top_places_by_incident_count.csv"), index=False)

# ============================================================
# 19. FINISH
# ============================================================
print("\nFinished.")
print("Outputs saved in:", OUTPUT_DIR)
print("\nMain Leaflet map files:")
print("01_event_points_map.html")
print("02_event_cluster_map.html")
print("03_event_concentration_heatmap.html")
print("04_severity_heatmap.html")
print("05_death_heatmap.html")
print("06_injury_heatmap.html")
print("07_cattle_stolen_heatmap.html")
print("08_abduction_heatmap.html")
print("09_displacement_heatmap.html")
print("10_bubble_map_deaths.html")
print("11_bubble_map_injuries.html")
print("12_bubble_map_cattle.html")
print("13_bubble_map_abductions.html")
print("14_bubble_map_displacement.html")
print("15_bubble_map_severity.html")
print("16_weapon_type_layered_map.html")
print("17_heatmap_year_<YEAR>.html")
print("18_county_concentration_map.html")
print("19_master_leaflet_map.html")