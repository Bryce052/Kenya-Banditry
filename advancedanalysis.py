# ============================================================
# ADVANCED SNA + GIS ANALYSIS IN ONE PYTHON FILE
# ============================================================
# PURPOSE
# This script performs advanced Social Network Analysis (SNA),
# co-occurrence analysis, severity analysis, clustering, and
# geospatial analysis for the incident dataset in one file.
#
# IMPORTANT
# 1. Replace FILE_NAME with your real CSV file name.
# 2. This script assumes the dataset columns were renamed into the
#    standardized names below.
# 3. GIS mapping works best if you have latitude/longitude columns.
#    If you do not, the script can geocode places approximately,
#    but that is slower and less reliable.
#
# OUTPUTS
# All outputs are saved in the folder: advanced_outputs
#
# ============================================================

import os
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx

from wordcloud import WordCloud

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# Optional GIS / mapping packages
GIS_AVAILABLE = True
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    GIS_AVAILABLE = False

FOLIUM_AVAILABLE = True
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
except Exception:
    FOLIUM_AVAILABLE = False

GEOPY_AVAILABLE = True
try:
    from geopy.geocoders import Nominatim
except Exception:
    GEOPY_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================
FILE_NAME = "kimanalysis.csv"   # <-- CHANGE THIS
OUTPUT_DIR = "advanced_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 7)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(FILE_NAME)

print("\nLoaded dataset shape:", df.shape)
print("\nOriginal columns:")
for c in df.columns:
    print("-", c)

# ============================================================
# 2. STANDARDIZE COLUMN NAMES BY POSITION
#    Based on the variable order you provided
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
for i, col in enumerate(df.columns[:len(expected_columns)]):
    rename_map[col] = expected_columns[i]
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
    if x.lower() in ["unknown", "unk", "n/a", "na", "none", "nil", "nan"]:
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

month_map = {
    "JAN": "JAN", "JANUARY": "JAN",
    "FEB": "FEB", "FEBRUARY": "FEB",
    "MAR": "MAR", "MARCH": "MAR",
    "APR": "APR", "APRIL": "APR",
    "MAY": "MAY",
    "JUN": "JUN", "JUNE": "JUN",
    "JUL": "JUL", "JULY": "JUL",
    "AUG": "AUG", "AUGUST": "AUG",
    "SEP": "SEP", "SEPT": "SEP", "SEPTEMBER": "SEP",
    "OCT": "OCT", "OCTOBER": "OCT",
    "NOV": "NOV", "NOVEMBER": "NOV",
    "DEC": "DEC", "DECEMBER": "DEC"
}
month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

if "month" in df.columns:
    df["month"] = df["month"].astype(str).str.upper().str.strip()
    df["month"] = df["month"].map(lambda x: month_map.get(x, x) if pd.notna(x) else np.nan)

for c in ["weapon_type", "weapon_details", "facility_attacked", "media_house",
          "other_crime_1", "other_crime_2", "other_crime_3", "ammunition_type"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.upper().str.strip()
        df[c] = df[c].replace({"NAN": np.nan})

# Ensure document code exists
if "doc_code" not in df.columns:
    df["doc_code"] = [f"DOC_{i+1:05d}" for i in range(len(df))]

# ============================================================
# 4. HELPERS
# ============================================================
def split_semicolon_values(value):
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).split(";") if v.strip()]

def safe_value(val):
    return 0 if pd.isna(val) else val

def save_series_csv(series, filename, index_name="category", value_name="count"):
    out = series.reset_index()
    out.columns = [index_name, value_name]
    out.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

def save_plot(fig_name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name), dpi=300)
    plt.close()

# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
other_crime_cols = [c for c in ["other_crime_1", "other_crime_2", "other_crime_3"] if c in df.columns]

df["other_crimes_list"] = df.apply(
    lambda row: [row[c] for c in other_crime_cols if c in row.index and pd.notna(row[c])],
    axis=1
)

if "facility_attacked" in df.columns:
    df["facility_list"] = df["facility_attacked"].apply(split_semicolon_values)
else:
    df["facility_list"] = [[] for _ in range(len(df))]

# simple severity index
# weights can be changed depending on your thesis logic
df["severity_index"] = (
    3 * df.get("total_deaths", pd.Series(0, index=df.index)).fillna(0)
    + 2 * df.get("total_injuries", pd.Series(0, index=df.index)).fillna(0)
    + 0.02 * df.get("cattle_stolen", pd.Series(0, index=df.index)).fillna(0)
    + 0.5 * df.get("people_abducted", pd.Series(0, index=df.index)).fillna(0)
    + 0.01 * df.get("people_displaced", pd.Series(0, index=df.index)).fillna(0)
    + 0.2 * df.get("num_attacking_bandits", pd.Series(0, index=df.index)).fillna(0)
)

# ============================================================
# 6. NETWORK ANALYSIS
# ============================================================
print("\nRunning network analysis...")

# ------------------------------------------------------------
# 6A. Incident bipartite graph
# ------------------------------------------------------------
B = nx.Graph()

for _, row in df.iterrows():
    incident = f"INCIDENT::{row['doc_code']}"
    B.add_node(incident, node_type="incident")

    if pd.notna(row.get("place_name")):
        place = f"PLACE::{row['place_name']}"
        B.add_node(place, node_type="place")
        B.add_edge(incident, place)

    if pd.notna(row.get("weapon_type")):
        weapon = f"WEAPON::{row['weapon_type']}"
        B.add_node(weapon, node_type="weapon")
        B.add_edge(incident, weapon)

    if pd.notna(row.get("media_house")):
        media = f"MEDIA::{row['media_house']}"
        B.add_node(media, node_type="media")
        B.add_edge(incident, media)

    for crime in row["other_crimes_list"]:
        crime_node = f"CRIME::{crime}"
        B.add_node(crime_node, node_type="crime")
        B.add_edge(incident, crime_node)

    for facility in row["facility_list"]:
        facility_node = f"FACILITY::{facility}"
        B.add_node(facility_node, node_type="facility")
        B.add_edge(incident, facility_node)

print("Bipartite nodes:", B.number_of_nodes())
print("Bipartite edges:", B.number_of_edges())

# save node/edge lists
b_nodes = pd.DataFrame([{"node": n, **B.nodes[n]} for n in B.nodes()])
b_edges = pd.DataFrame([{"source": u, "target": v} for u, v in B.edges()])
b_nodes.to_csv(os.path.join(OUTPUT_DIR, "bipartite_nodes.csv"), index=False)
b_edges.to_csv(os.path.join(OUTPUT_DIR, "bipartite_edges.csv"), index=False)

# ------------------------------------------------------------
# 6B. Crime-weapon-place co-occurrence network
# ------------------------------------------------------------
G = nx.Graph()

def add_weighted_edge(graph, a, b):
    if graph.has_edge(a, b):
        graph[a][b]["weight"] += 1
    else:
        graph.add_edge(a, b, weight=1)

for _, row in df.iterrows():
    entities = []

    if pd.notna(row.get("place_name")):
        entities.append(f"PLACE::{row['place_name']}")
    if pd.notna(row.get("weapon_type")):
        entities.append(f"WEAPON::{row['weapon_type']}")
    if pd.notna(row.get("media_house")):
        entities.append(f"MEDIA::{row['media_house']}")

    entities.extend([f"CRIME::{x}" for x in row["other_crimes_list"]])
    entities.extend([f"FACILITY::{x}" for x in row["facility_list"]])

    entities = list(set(entities))

    for e in entities:
        if e not in G:
            prefix = e.split("::")[0]
            G.add_node(e, node_type=prefix.lower())

    for a, b in combinations(entities, 2):
        add_weighted_edge(G, a, b)

print("Co-occurrence graph nodes:", G.number_of_nodes())
print("Co-occurrence graph edges:", G.number_of_edges())

# centrality
if G.number_of_nodes() > 0:
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G, weight="weight")

    centrality_df = pd.DataFrame({
        "node": list(G.nodes()),
        "node_type": [G.nodes[n].get("node_type") for n in G.nodes()],
        "degree_centrality": [degree_cent[n] for n in G.nodes()],
        "betweenness_centrality": [between_cent[n] for n in G.nodes()]
    }).sort_values("degree_centrality", ascending=False)

    centrality_df.to_csv(os.path.join(OUTPUT_DIR, "network_centrality.csv"), index=False)

    print("\nTop 15 central nodes:")
    print(centrality_df.head(15))

# filtered graph visualization
top_nodes = []
if G.number_of_nodes() > 0:
    deg = dict(G.degree(weight="weight"))
    top_nodes = sorted(deg, key=deg.get, reverse=True)[:40]
    H = G.subgraph(top_nodes).copy()

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(H, seed=42, k=0.5)

    sizes = [300 + 2500 * nx.degree_centrality(H)[n] for n in H.nodes()]
    labels = {n: n.split("::", 1)[1] for n in H.nodes()}

    nx.draw_networkx_nodes(H, pos, node_size=sizes)
    nx.draw_networkx_edges(H, pos, alpha=0.4, width=1.2)
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

    plt.title("Top Co-Occurrence Network")
    plt.axis("off")
    save_plot("network_top_cooccurrence.png")

# ------------------------------------------------------------
# 6C. Place-weapon network
# ------------------------------------------------------------
PW = nx.Graph()
for _, row in df.iterrows():
    place = row.get("place_name")
    weapon = row.get("weapon_type")
    if pd.notna(place) and pd.notna(weapon):
        add_weighted_edge(PW, f"PLACE::{place}", f"WEAPON::{weapon}")

if PW.number_of_nodes() > 0:
    pw_edges = pd.DataFrame([
        {"source": u, "target": v, "weight": d["weight"]}
        for u, v, d in PW.edges(data=True)
    ]).sort_values("weight", ascending=False)
    pw_edges.to_csv(os.path.join(OUTPUT_DIR, "place_weapon_edges.csv"), index=False)

    top_pw = pw_edges.head(30)
    plt.figure(figsize=(12, 8))
    plt.barh(
        y=[f"{r['source'].split('::')[1]} ↔ {r['target'].split('::')[1]}" for _, r in top_pw[::-1].iterrows()],
        width=top_pw[::-1]["weight"]
    )
    plt.title("Top Place-Weapon Links")
    plt.xlabel("Co-occurrence Count")
    save_plot("top_place_weapon_links.png")

# ------------------------------------------------------------
# 6D. Crime co-occurrence matrix
# ------------------------------------------------------------
all_crimes = sorted(
    list(
        set(
            crime
            for crimes in df["other_crimes_list"]
            for crime in crimes
        )
    )
)

if len(all_crimes) > 0:
    crime_matrix = pd.DataFrame(0, index=all_crimes, columns=all_crimes)

    for crimes in df["other_crimes_list"]:
        unique_crimes = list(set(crimes))
        for a in unique_crimes:
            for b in unique_crimes:
                if a != b:
                    crime_matrix.loc[a, b] += 1

    crime_matrix.to_csv(os.path.join(OUTPUT_DIR, "crime_cooccurrence_matrix.csv"))

    plt.figure(figsize=(12, 10))
    plt.imshow(crime_matrix, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(crime_matrix.columns)), crime_matrix.columns, rotation=90)
    plt.yticks(range(len(crime_matrix.index)), crime_matrix.index)
    plt.title("Crime Co-Occurrence Matrix")
    save_plot("crime_cooccurrence_matrix.png")

# ------------------------------------------------------------
# 6E. Sankey diagram: Weapon -> Crime -> Outcome
# ------------------------------------------------------------
sankey_rows = []
for _, row in df.iterrows():
    weapon = row.get("weapon_type")
    crimes = row["other_crimes_list"]
    deaths = safe_value(row.get("total_deaths"))
    injuries = safe_value(row.get("total_injuries"))

    if pd.notna(weapon):
        if len(crimes) == 0:
            crimes = ["NO_OTHER_CRIME"]

        for crime in crimes:
            outcome = "HIGH_FATALITY" if deaths >= 10 else "HIGH_INJURY" if injuries >= 10 else "LOWER_SEVERITY"
            sankey_rows.append((weapon, crime, outcome))

if len(sankey_rows) > 0:
    sankey_df = pd.DataFrame(sankey_rows, columns=["weapon", "crime", "outcome"])
    sankey_df.to_csv(os.path.join(OUTPUT_DIR, "sankey_source_data.csv"), index=False)

    # build sankey
    labels = list(pd.unique(sankey_df[["weapon", "crime", "outcome"]].values.ravel()))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    src, tgt, val = [], [], []

    weapon_crime = sankey_df.groupby(["weapon", "crime"]).size().reset_index(name="count")
    for _, r in weapon_crime.iterrows():
        src.append(label_to_idx[r["weapon"]])
        tgt.append(label_to_idx[r["crime"]])
        val.append(r["count"])

    crime_outcome = sankey_df.groupby(["crime", "outcome"]).size().reset_index(name="count")
    for _, r in crime_outcome.iterrows():
        src.append(label_to_idx[r["crime"]])
        tgt.append(label_to_idx[r["outcome"]])
        val.append(r["count"])

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=15),
        link=dict(source=src, target=tgt, value=val)
    )])
    fig.update_layout(title_text="Weapon → Crime → Outcome Sankey", font_size=10)
    fig.write_html(os.path.join(OUTPUT_DIR, "weapon_crime_outcome_sankey.html"))

# ============================================================
# 7. CLUSTERING / INCIDENT TYPOLOGY
# ============================================================
print("\nRunning clustering...")

cluster_features = [
    c for c in [
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
        "num_weapons",
        "severity_index"
    ] if c in df.columns
]

X_num = df[cluster_features].fillna(0).copy()

# add selected categorical indicators
cat_lists = []
for col in ["weapon_type", "media_house"]:
    if col in df.columns:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
        if dummies.shape[1] > 0:
            cat_lists.append(dummies)

if len(df["other_crimes_list"]) > 0:
    mlb_crime = MultiLabelBinarizer()
    crime_bin = pd.DataFrame(
        mlb_crime.fit_transform(df["other_crimes_list"]),
        columns=[f"crime_{c}" for c in mlb_crime.classes_],
        index=df.index
    )
    if crime_bin.shape[1] > 0:
        cat_lists.append(crime_bin)

if len(df["facility_list"]) > 0:
    mlb_fac = MultiLabelBinarizer()
    fac_bin = pd.DataFrame(
        mlb_fac.fit_transform(df["facility_list"]),
        columns=[f"facility_{c}" for c in mlb_fac.classes_],
        index=df.index
    )
    if fac_bin.shape[1] > 0:
        cat_lists.append(fac_bin)

X = pd.concat([X_num] + cat_lists, axis=1).fillna(0)

if X.shape[0] >= 5 and X.shape[1] >= 2:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = 4 if X.shape[0] >= 8 else 3
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_summary = df.groupby("cluster")[cluster_features].mean().round(2)
    cluster_summary["n_incidents"] = df["cluster"].value_counts().sort_index()
    cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"))

    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster": df["cluster"].astype(str),
        "doc_code": df["doc_code"].astype(str),
        "severity_index": df["severity_index"]
    })
    pca_df.to_csv(os.path.join(OUTPUT_DIR, "incident_pca_projection.csv"), index=False)

    plt.figure(figsize=(10, 8))
    for cl in sorted(df["cluster"].dropna().unique()):
        subset = pca_df[pca_df["cluster"] == str(cl)]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cl}", alpha=0.7)

    plt.title("Incident Typology Clusters (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    save_plot("incident_clusters_pca.png")

# ============================================================
# 8. GEOSPATIAL ANALYSIS
# ============================================================
print("\nRunning GIS analysis...")

# ------------------------------------------------------------
# 8A. If latitude/longitude already exist, use them
# ------------------------------------------------------------
lat_candidates = [c for c in df.columns if c.lower() in ["latitude", "lat", "y"]]
lon_candidates = [c for c in df.columns if c.lower() in ["longitude", "lon", "lng", "long", "x"]]

if len(lat_candidates) > 0 and len(lon_candidates) > 0:
    lat_col = lat_candidates[0]
    lon_col = lon_candidates[0]
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
else:
    # --------------------------------------------------------
    # 8B. Optional geocoding from place_name
    # --------------------------------------------------------
    # This only runs if no coordinates exist
    # It uses a simple cache so repeated place names are only geocoded once
    if GEOPY_AVAILABLE and "place_name" in df.columns:
        geocode_cache = {}
        geolocator = Nominatim(user_agent="incident_analysis_geocoder")

        unique_places = df["place_name"].dropna().astype(str).unique().tolist()
        latitudes = {}
        longitudes = {}

        print("No coordinate columns found. Attempting approximate geocoding from place_name...")

        for place in unique_places:
            try:
                query = f"{place}, Kenya"
                loc = geolocator.geocode(query, timeout=10)
                if loc is not None:
                    latitudes[place] = loc.latitude
                    longitudes[place] = loc.longitude
                else:
                    latitudes[place] = np.nan
                    longitudes[place] = np.nan
            except Exception:
                latitudes[place] = np.nan
                longitudes[place] = np.nan

        df["latitude"] = df["place_name"].map(latitudes)
        df["longitude"] = df["place_name"].map(longitudes)
    else:
        df["latitude"] = np.nan
        df["longitude"] = np.nan

df[["doc_code", "place_name", "latitude", "longitude"]].to_csv(
    os.path.join(OUTPUT_DIR, "geocoded_places.csv"), index=False
)

geo_df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
print("Rows with usable coordinates:", len(geo_df))

# ------------------------------------------------------------
# 8C. Static point map
# ------------------------------------------------------------
if len(geo_df) > 0 and GIS_AVAILABLE:
    gdf = gpd.GeoDataFrame(
        geo_df,
        geometry=[Point(xy) for xy in zip(geo_df["longitude"], geo_df["latitude"])],
        crs="EPSG:4326"
    )

    gdf.to_file(os.path.join(OUTPUT_DIR, "incident_points.geojson"), driver="GeoJSON")

    plt.figure(figsize=(10, 10))
    plt.scatter(gdf["longitude"], gdf["latitude"], s=10 + gdf["severity_index"], alpha=0.6)
    plt.title("Incident Point Distribution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    save_plot("incident_point_distribution.png")

# ------------------------------------------------------------
# 8D. Interactive folium map
# ------------------------------------------------------------
if len(geo_df) > 0 and FOLIUM_AVAILABLE:
    center_lat = geo_df["latitude"].mean()
    center_lon = geo_df["longitude"].mean()

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    marker_cluster = MarkerCluster().add_to(fmap)

    for _, row in geo_df.iterrows():
        popup = (
            f"<b>Doc Code:</b> {row.get('doc_code', '')}<br>"
            f"<b>Year:</b> {row.get('year', '')}<br>"
            f"<b>Month:</b> {row.get('month', '')}<br>"
            f"<b>Place:</b> {row.get('place_name', '')}<br>"
            f"<b>Deaths:</b> {row.get('total_deaths', '')}<br>"
            f"<b>Injuries:</b> {row.get('total_injuries', '')}<br>"
            f"<b>Severity:</b> {round(row.get('severity_index', 0), 2)}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(4, min(15, 3 + row["severity_index"] / 5)),
            popup=popup,
            fill=True
        ).add_to(marker_cluster)

    fmap.save(os.path.join(OUTPUT_DIR, "interactive_incident_map.html"))

# ------------------------------------------------------------
# 8E. Heat map
# ------------------------------------------------------------
if len(geo_df) > 0 and FOLIUM_AVAILABLE:
    heat_data = geo_df[["latitude", "longitude", "severity_index"]].fillna(0).values.tolist()
    heat_map = folium.Map(location=[geo_df["latitude"].mean(), geo_df["longitude"].mean()], zoom_start=6)
    HeatMap(heat_data, radius=15, blur=10, max_zoom=8).add_to(heat_map)
    heat_map.save(os.path.join(OUTPUT_DIR, "incident_heatmap.html"))

# ------------------------------------------------------------
# 8F. Severity by place
# ------------------------------------------------------------
if "place_name" in df.columns:
    place_severity = df.groupby("place_name").agg(
        incidents=("doc_code", "count"),
        mean_severity=("severity_index", "mean"),
        total_deaths=("total_deaths", "sum"),
        total_injuries=("total_injuries", "sum")
    ).sort_values("mean_severity", ascending=False)

    place_severity.to_csv(os.path.join(OUTPUT_DIR, "place_severity_summary.csv"))

    plt.figure(figsize=(12, 8))
    place_severity.head(20).sort_values("mean_severity").plot(
        y="mean_severity", kind="barh", legend=False
    )
    plt.title("Top 20 Places by Mean Severity")
    plt.xlabel("Mean Severity Index")
    save_plot("top_places_mean_severity.png")

# ============================================================
# 9. SPATIO-TEMPORAL ANALYSIS
# ============================================================
print("\nRunning spatio-temporal analysis...")

if "year" in df.columns:
    yearly = df.groupby("year").agg(
        incidents=("doc_code", "count"),
        total_deaths=("total_deaths", "sum"),
        total_injuries=("total_injuries", "sum"),
        avg_severity=("severity_index", "mean")
    )
    yearly.to_csv(os.path.join(OUTPUT_DIR, "yearly_spatiotemporal_summary.csv"))

    plt.figure()
    yearly["incidents"].plot(marker="o")
    plt.title("Incidents by Year")
    plt.xlabel("Year")
    plt.ylabel("Incidents")
    save_plot("incidents_by_year_advanced.png")

    plt.figure()
    yearly["avg_severity"].plot(marker="o")
    plt.title("Average Severity by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Severity")
    save_plot("avg_severity_by_year.png")

if "month" in df.columns:
    monthly = df.groupby("month").agg(
        incidents=("doc_code", "count"),
        total_deaths=("total_deaths", "sum"),
        total_injuries=("total_injuries", "sum"),
        avg_severity=("severity_index", "mean")
    ).reindex(month_order)
    monthly.to_csv(os.path.join(OUTPUT_DIR, "monthly_spatiotemporal_summary.csv"))

    plt.figure()
    monthly["incidents"].plot(marker="o")
    plt.title("Incidents by Month")
    plt.xlabel("Month")
    plt.ylabel("Incidents")
    save_plot("incidents_by_month_advanced.png")

# year-place matrix
if "year" in df.columns and "place_name" in df.columns:
    year_place = pd.crosstab(df["year"], df["place_name"])
    year_place.to_csv(os.path.join(OUTPUT_DIR, "year_place_matrix.csv"))

    top_places = df["place_name"].value_counts().head(15).index
    yp_small = pd.crosstab(df["year"], df["place_name"])[top_places]
    plt.figure(figsize=(14, 8))
    plt.imshow(yp_small.T, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(yp_small.index)), yp_small.index, rotation=45)
    plt.yticks(range(len(yp_small.columns)), yp_small.columns)
    plt.title("Top Place Activity Over Time")
    save_plot("top_place_activity_heatmap.png")

# ============================================================
# 10. WORD CLOUDS FOR ADVANCED CATEGORIES
# ============================================================
if "comments" in df.columns:
    comments_text = " ".join(df["comments"].dropna().astype(str))
    if comments_text.strip():
        wc = WordCloud(width=1400, height=700, background_color="white").generate(comments_text)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Comments Word Cloud")
        save_plot("comments_wordcloud.png")

if "media_house" in df.columns:
    media_text = " ".join(df["media_house"].dropna().astype(str))
    if media_text.strip():
        wc = WordCloud(width=1400, height=700, background_color="white").generate(media_text)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Media House Word Cloud")
        save_plot("media_house_wordcloud.png")

# ============================================================
# 11. EXPORT CLEAN DATA + TOP TABLES
# ============================================================
df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_advanced_dataset.csv"), index=False)

if "severity_index" in df.columns:
    top_severe = df.sort_values("severity_index", ascending=False)[[
        c for c in [
            "doc_code", "year", "month", "place_name", "weapon_type",
            "total_deaths", "total_injuries", "people_displaced",
            "people_abducted", "severity_index"
        ] if c in df.columns
    ]].head(30)
    top_severe.to_csv(os.path.join(OUTPUT_DIR, "top_30_most_severe_incidents.csv"), index=False)

# ============================================================
# 12. EXECUTIVE OUTPUT SUMMARY
# ============================================================
summary = pd.DataFrame([
    ["rows", len(df)],
    ["columns", df.shape[1]],
    ["unique_doc_codes", df["doc_code"].nunique(dropna=True)],
    ["network_nodes", G.number_of_nodes()],
    ["network_edges", G.number_of_edges()],
    ["bipartite_nodes", B.number_of_nodes()],
    ["bipartite_edges", B.number_of_edges()],
    ["rows_with_coordinates", len(geo_df)],
    ["mean_severity", round(df["severity_index"].mean(), 3)],
    ["max_severity", round(df["severity_index"].max(), 3)]
], columns=["metric", "value"])

summary.to_csv(os.path.join(OUTPUT_DIR, "advanced_analysis_summary.csv"), index=False)

print("\nDone.")
print("\nOutputs saved to:", OUTPUT_DIR)
print("\nMain outputs:")
print("- cleaned_advanced_dataset.csv")
print("- advanced_analysis_summary.csv")
print("- network_centrality.csv")
print("- bipartite_nodes.csv / bipartite_edges.csv")
print("- place_weapon_edges.csv")
print("- crime_cooccurrence_matrix.csv")
print("- weapon_crime_outcome_sankey.html")
print("- cluster_summary.csv")
print("- incident_clusters_pca.png")
print("- geocoded_places.csv")
print("- incident_points.geojson")
print("- interactive_incident_map.html")
print("- incident_heatmap.html")
print("- top_places_mean_severity.png")
print("- incidents_by_year_advanced.png")
print("- top_place_activity_heatmap.png")