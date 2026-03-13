import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
FILE_NAME = "kimanalysis.csv"   # <-- replace with your actual file name
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(FILE_NAME)

print("\n==============================")
print("RAW DATA PREVIEW")
print("==============================")
print(df.head())
print("\nRaw shape:", df.shape)
print("\nRaw columns:")
for i, c in enumerate(df.columns):
    print(f"{i}: {c}")

# ============================================================
# 2. STANDARDIZE / RENAME COLUMNS
#    Uses the exact variable order you pasted
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

# rename the first N columns by position, if possible
rename_by_position = {}
for i, col in enumerate(df.columns[:len(expected_columns)]):
    rename_by_position[col] = expected_columns[i]

df = df.rename(columns=rename_by_position)

# if there are extra unnamed columns, keep them but label them clearly
extra_cols = df.columns[len(expected_columns):]
for i, c in enumerate(extra_cols, start=1):
    if "Unnamed" in str(c) or str(c).strip() == "":
        df = df.rename(columns={c: f"extra_col_{i}"})

print("\n==============================")
print("RENAMED / STANDARDIZED COLUMNS")
print("==============================")
for c in df.columns:
    print(c)

# ============================================================
# 3. BASIC CLEANING
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

# clean object columns
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].apply(clean_text)

# month normalization
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

if "month" in df.columns:
    df["month"] = df["month"].astype(str).str.upper().str.strip()
    df["month"] = df["month"].map(lambda x: month_map.get(x, x) if pd.notna(x) else np.nan)

month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# numeric columns
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

# standardize some text columns to uppercase for consistency
for c in ["weapon_type", "ammunition_type", "media_house", "facility_attacked"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.upper().str.strip()
        df[c] = df[c].replace({"NAN": np.nan})

print("\n==============================")
print("CLEANED DATA PREVIEW")
print("==============================")
print(df.head())
print("\nCleaned shape:", df.shape)

# ============================================================
# 4. DATA QUALITY / DOCUMENT CODE CHECK
# ============================================================
print("\n==============================")
print("DOCUMENT CODE CHECK")
print("==============================")
if "doc_code" in df.columns:
    print("Non-null document codes:", df["doc_code"].notna().sum())
    print("Unique document codes:", df["doc_code"].nunique(dropna=True))
    print("Sample document codes:")
    print(df["doc_code"].dropna().head(10).tolist())

missing_summary = df.isna().sum().sort_values(ascending=False)
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
quality_table = pd.DataFrame({
    "missing_count": missing_summary,
    "missing_percent": missing_pct.round(2),
    "dtype": df.dtypes.astype(str)
})
quality_table.to_csv(os.path.join(OUTPUT_DIR, "data_quality_summary.csv"))

print("\n==============================")
print("MISSING VALUES SUMMARY")
print("==============================")
print(quality_table)

# plot missing values
plt.figure(figsize=(14, 7))
missing_pct.sort_values(ascending=False).plot(kind="bar")
plt.title("Missing Values by Variable (%)")
plt.ylabel("Percent Missing")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "missing_values_percent.png"), dpi=300)
plt.close()

# ============================================================
# 5. DESCRIPTIVE STATISTICS FOR NUMERIC VARIABLES
# ============================================================
available_numeric = [c for c in numeric_cols if c in df.columns]
desc_stats = df[available_numeric].describe().T
desc_stats["missing"] = df[available_numeric].isna().sum()
desc_stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics_numeric.csv"))

print("\n==============================")
print("DESCRIPTIVE STATISTICS - NUMERIC VARIABLES")
print("==============================")
print(desc_stats)

# histograms for each numeric variable
for c in available_numeric:
    plt.figure()
    df[c].dropna().plot(kind="hist", bins=20)
    plt.title(f"Distribution of {c}")
    plt.xlabel(c)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{c}.png"), dpi=300)
    plt.close()

# boxplots for each numeric variable
for c in available_numeric:
    plt.figure()
    plt.boxplot(df[c].dropna(), vert=True)
    plt.title(f"Boxplot of {c}")
    plt.ylabel(c)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{c}.png"), dpi=300)
    plt.close()

# ============================================================
# 6. YEAR ANALYSIS
# ============================================================
if "year" in df.columns:
    year_counts = df["year"].value_counts(dropna=True).sort_index()
    year_counts.to_csv(os.path.join(OUTPUT_DIR, "incidents_by_year.csv"))

    plt.figure()
    year_counts.plot(kind="bar")
    plt.title("Number of Incidents by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Incidents")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "incidents_by_year.png"), dpi=300)
    plt.close()

    # year trends for key outcomes
    yearly_metrics = df.groupby("year")[[
        c for c in [
            "total_deaths", "police_deaths", "local_deaths", "suspected_bandits_killed",
            "total_injuries", "police_injured", "locals_injured", "bandits_injured",
            "people_displaced", "people_abducted", "cattle_stolen", "num_attacking_bandits"
        ] if c in df.columns
    ]].sum(min_count=1)

    yearly_metrics.to_csv(os.path.join(OUTPUT_DIR, "yearly_metrics_sum.csv"))

    for c in yearly_metrics.columns:
        plt.figure()
        yearly_metrics[c].plot(marker="o")
        plt.title(f"Yearly Trend: {c}")
        plt.xlabel("Year")
        plt.ylabel(c)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"yearly_trend_{c}.png"), dpi=300)
        plt.close()

# ============================================================
# 7. MONTH ANALYSIS
# ============================================================
if "month" in df.columns:
    month_counts = df["month"].value_counts().reindex(month_order)
    month_counts.to_csv(os.path.join(OUTPUT_DIR, "incidents_by_month.csv"))

    plt.figure()
    month_counts.plot(kind="bar")
    plt.title("Number of Incidents by Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Incidents")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "incidents_by_month.png"), dpi=300)
    plt.close()

    monthly_metrics = df.groupby("month")[[
        c for c in [
            "total_deaths", "total_injuries", "people_displaced",
            "people_abducted", "cattle_stolen", "num_attacking_bandits"
        ] if c in df.columns
    ]].sum(min_count=1).reindex(month_order)

    monthly_metrics.to_csv(os.path.join(OUTPUT_DIR, "monthly_metrics_sum.csv"))

    for c in monthly_metrics.columns:
        plt.figure()
        monthly_metrics[c].plot(marker="o")
        plt.title(f"Monthly Trend: {c}")
        plt.xlabel("Month")
        plt.ylabel(c)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"monthly_trend_{c}.png"), dpi=300)
        plt.close()

# ============================================================
# 8. PLACE / GEOGRAPHIC DISTRIBUTION
# ============================================================
if "place_name" in df.columns:
    place_counts = df["place_name"].value_counts().head(25)
    place_counts.to_csv(os.path.join(OUTPUT_DIR, "top_places_by_frequency.csv"))

    plt.figure(figsize=(12, 8))
    place_counts.sort_values().plot(kind="barh")
    plt.title("Top 25 Places by Incident Frequency")
    plt.xlabel("Number of Incidents")
    plt.ylabel("Place")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_places_by_frequency.png"), dpi=300)
    plt.close()

    # explode semicolon-separated counties / sub-counties if present
    split_places = (
        df["place_name"]
        .dropna()
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )
    split_place_counts = split_places.value_counts().head(30)
    split_place_counts.to_csv(os.path.join(OUTPUT_DIR, "split_place_counts.csv"))

    plt.figure(figsize=(12, 8))
    split_place_counts.sort_values().plot(kind="barh")
    plt.title("Top Split Place Mentions")
    plt.xlabel("Frequency")
    plt.ylabel("Place Component")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "split_place_mentions.png"), dpi=300)
    plt.close()

# ============================================================
# 9. DEATHS ANALYSIS
# ============================================================
death_cols = [c for c in ["total_deaths", "police_deaths", "local_deaths", "suspected_bandits_killed"] if c in df.columns]
if death_cols:
    death_summary = df[death_cols].describe().T
    death_summary["sum"] = df[death_cols].sum()
    death_summary.to_csv(os.path.join(OUTPUT_DIR, "deaths_summary.csv"))

    if set(["police_deaths", "local_deaths", "suspected_bandits_killed"]).issubset(df.columns):
        death_totals = df[["police_deaths", "local_deaths", "suspected_bandits_killed"]].sum()
        plt.figure()
        death_totals.plot(kind="bar")
        plt.title("Death Composition by Group")
        plt.ylabel("Total Deaths")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "death_composition.png"), dpi=300)
        plt.close()

# ============================================================
# 10. INJURY ANALYSIS
# ============================================================
injury_cols = [c for c in ["total_injuries", "police_injured", "locals_injured", "bandits_injured"] if c in df.columns]
if injury_cols:
    injury_summary = df[injury_cols].describe().T
    injury_summary["sum"] = df[injury_cols].sum()
    injury_summary.to_csv(os.path.join(OUTPUT_DIR, "injuries_summary.csv"))

    if set(["police_injured", "locals_injured", "bandits_injured"]).issubset(df.columns):
        injury_totals = df[["police_injured", "locals_injured", "bandits_injured"]].sum()
        plt.figure()
        injury_totals.plot(kind="bar")
        plt.title("Injury Composition by Group")
        plt.ylabel("Total Injuries")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "injury_composition.png"), dpi=300)
        plt.close()

# ============================================================
# 11. CATTLE STOLEN / BANDITS / DISPLACEMENT / ABDUCTION
# ============================================================
for c in ["cattle_stolen", "num_attacking_bandits", "people_displaced", "people_abducted", "num_weapons"]:
    if c in df.columns:
        summary = df[c].describe()
        summary.to_csv(os.path.join(OUTPUT_DIR, f"summary_{c}.csv"))

for c in ["cattle_stolen", "num_attacking_bandits", "people_displaced", "people_abducted"]:
    if c in df.columns:
        plt.figure()
        df[c].dropna().plot(kind="hist", bins=20)
        plt.title(f"Distribution of {c}")
        plt.xlabel(c)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"distribution_{c}.png"), dpi=300)
        plt.close()

# ============================================================
# 12. OTHER CRIMES ANALYSIS
# ============================================================
other_crime_cols = [c for c in ["other_crime_1", "other_crime_2", "other_crime_3"] if c in df.columns]

if other_crime_cols:
    all_other_crimes = pd.Series(dtype=object)
    for c in other_crime_cols:
        s = df[c].dropna().astype(str).str.upper().str.strip()
        all_other_crimes = pd.concat([all_other_crimes, s], ignore_index=True)

    if len(all_other_crimes) > 0:
        other_crime_counts = all_other_crimes.value_counts()
        other_crime_counts.to_csv(os.path.join(OUTPUT_DIR, "other_crimes_frequency.csv"))

        plt.figure(figsize=(12, 8))
        other_crime_counts.head(20).sort_values().plot(kind="barh")
        plt.title("Top Other Crimes")
        plt.xlabel("Frequency")
        plt.ylabel("Crime Type")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "other_crimes_frequency.png"), dpi=300)
        plt.close()

        # word cloud
        text = " ".join(all_other_crimes.dropna().astype(str))
        wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Other Crimes")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "other_crimes_wordcloud.png"), dpi=300)
        plt.close()

# ============================================================
# 13. WEAPON ANALYSIS
# ============================================================
if "weapon_type" in df.columns:
    weapon_counts = df["weapon_type"].dropna().value_counts()
    weapon_counts.to_csv(os.path.join(OUTPUT_DIR, "weapon_type_frequency.csv"))

    plt.figure()
    weapon_counts.plot(kind="bar")
    plt.title("Weapon Type Frequency")
    plt.xlabel("Weapon Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "weapon_type_frequency.png"), dpi=300)
    plt.close()

if "weapon_details" in df.columns:
    details = df["weapon_details"].dropna().astype(str)
    if len(details) > 0:
        details_counts = details.value_counts().head(30)
        details_counts.to_csv(os.path.join(OUTPUT_DIR, "weapon_details_frequency.csv"))

        plt.figure(figsize=(12, 8))
        details_counts.sort_values().plot(kind="barh")
        plt.title("Top Weapon Details")
        plt.xlabel("Frequency")
        plt.ylabel("Weapon Detail")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "weapon_details_frequency.png"), dpi=300)
        plt.close()

        wc = WordCloud(width=1200, height=600, background_color="white").generate(" ".join(details))
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Weapon Details")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "weapon_details_wordcloud.png"), dpi=300)
        plt.close()

if "firearm_ammunition_present" in df.columns:
    ammo_presence = df["firearm_ammunition_present"].dropna().astype(str).str.upper().value_counts()
    ammo_presence.to_csv(os.path.join(OUTPUT_DIR, "ammo_presence_frequency.csv"))

if "ammunition_type" in df.columns:
    ammo_type_counts = df["ammunition_type"].dropna().astype(str).str.upper().value_counts()
    ammo_type_counts.to_csv(os.path.join(OUTPUT_DIR, "ammunition_type_frequency.csv"))

    plt.figure()
    ammo_type_counts.plot(kind="bar")
    plt.title("Ammunition Type Frequency")
    plt.xlabel("Ammunition Type")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ammunition_type_frequency.png"), dpi=300)
    plt.close()

# ============================================================
# 14. FACILITY ATTACKED ANALYSIS
# ============================================================
if "facility_attacked" in df.columns:
    facilities = (
        df["facility_attacked"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.split(";")
        .explode()
        .str.strip()
    )
    facilities = facilities[facilities.notna() & (facilities != "")]
    if len(facilities) > 0:
        facility_counts = facilities.value_counts()
        facility_counts.to_csv(os.path.join(OUTPUT_DIR, "facility_attacked_frequency.csv"))

        plt.figure()
        facility_counts.plot(kind="bar")
        plt.title("Facility Types Attacked")
        plt.xlabel("Facility Type")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "facility_attacked_frequency.png"), dpi=300)
        plt.close()

# ============================================================
# 15. MEDIA HOUSE / SOURCE ANALYSIS
# ============================================================
if "media_house" in df.columns:
    media_counts = df["media_house"].dropna().value_counts()
    media_counts.to_csv(os.path.join(OUTPUT_DIR, "media_house_frequency.csv"))

    plt.figure(figsize=(12, 8))
    media_counts.head(20).sort_values().plot(kind="barh")
    plt.title("Top 20 Media Houses")
    plt.xlabel("Frequency")
    plt.ylabel("Media House")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "media_house_frequency.png"), dpi=300)
    plt.close()

    wc = WordCloud(width=1200, height=600, background_color="white").generate(" ".join(df["media_house"].dropna().astype(str)))
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Media Houses")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "media_house_wordcloud.png"), dpi=300)
    plt.close()

if "weblink" in df.columns:
    link_non_null = df["weblink"].notna().sum()
    link_summary = pd.DataFrame({
        "metric": ["total_rows", "rows_with_weblink", "rows_without_weblink"],
        "value": [len(df), link_non_null, len(df) - link_non_null]
    })
    link_summary.to_csv(os.path.join(OUTPUT_DIR, "weblink_summary.csv"), index=False)

# ============================================================
# 16. COMMENTS ANALYSIS
# ============================================================
if "comments" in df.columns:
    comments_non_null = df["comments"].dropna()
    comment_summary = pd.DataFrame({
        "metric": ["rows_with_comments", "rows_without_comments"],
        "value": [comments_non_null.shape[0], len(df) - comments_non_null.shape[0]]
    })
    comment_summary.to_csv(os.path.join(OUTPUT_DIR, "comments_summary.csv"), index=False)

    if len(comments_non_null) > 0:
        wc = WordCloud(width=1200, height=600, background_color="white").generate(" ".join(comments_non_null.astype(str)))
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Comments")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "comments_wordcloud.png"), dpi=300)
        plt.close()

# ============================================================
# 17. CORRELATION ANALYSIS FOR NUMERIC VARIABLES
# ============================================================
corr_vars = [c for c in available_numeric if df[c].notna().sum() > 1]
if len(corr_vars) >= 2:
    corr = df[corr_vars].corr(numeric_only=True)
    corr.to_csv(os.path.join(OUTPUT_DIR, "numeric_correlation_matrix.csv"))

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix of Numeric Variables")

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "numeric_correlation_matrix.png"), dpi=300)
    plt.close()

# ============================================================
# 18. CROSS-TAB ANALYSIS
# ============================================================
# year x month
if "year" in df.columns and "month" in df.columns:
    year_month_table = pd.crosstab(df["year"], df["month"]).reindex(columns=month_order)
    year_month_table.to_csv(os.path.join(OUTPUT_DIR, "crosstab_year_month.csv"))

# year x weapon type
if "year" in df.columns and "weapon_type" in df.columns:
    year_weapon = pd.crosstab(df["year"], df["weapon_type"])
    year_weapon.to_csv(os.path.join(OUTPUT_DIR, "crosstab_year_weapon_type.csv"))

# year x media house
if "year" in df.columns and "media_house" in df.columns:
    year_media = pd.crosstab(df["year"], df["media_house"])
    year_media.to_csv(os.path.join(OUTPUT_DIR, "crosstab_year_media_house.csv"))

# ============================================================
# 19. TOP INCIDENTS BY SEVERITY
# ============================================================
severity_cols = [c for c in ["doc_code", "year", "month", "place_name", "total_deaths", "total_injuries", "people_displaced", "people_abducted"] if c in df.columns]
if "total_deaths" in df.columns:
    top_deaths = df[severity_cols].sort_values("total_deaths", ascending=False).head(20)
    top_deaths.to_csv(os.path.join(OUTPUT_DIR, "top_20_incidents_by_deaths.csv"), index=False)

if "total_injuries" in df.columns:
    top_injuries = df[severity_cols].sort_values("total_injuries", ascending=False).head(20)
    top_injuries.to_csv(os.path.join(OUTPUT_DIR, "top_20_incidents_by_injuries.csv"), index=False)

if "people_displaced" in df.columns:
    top_displaced = df[severity_cols].sort_values("people_displaced", ascending=False).head(20)
    top_displaced.to_csv(os.path.join(OUTPUT_DIR, "top_20_incidents_by_displacement.csv"), index=False)

# ============================================================
# 20. EXECUTIVE SUMMARY TABLE
# ============================================================
summary_rows = []

summary_rows.append(["n_rows", len(df)])
summary_rows.append(["n_columns", df.shape[1]])

if "doc_code" in df.columns:
    summary_rows.append(["unique_doc_codes", df["doc_code"].nunique(dropna=True)])
if "year" in df.columns:
    summary_rows.append(["year_min", df["year"].min()])
    summary_rows.append(["year_max", df["year"].max()])
if "total_deaths" in df.columns:
    summary_rows.append(["sum_total_deaths", df["total_deaths"].sum()])
if "total_injuries" in df.columns:
    summary_rows.append(["sum_total_injuries", df["total_injuries"].sum()])
if "cattle_stolen" in df.columns:
    summary_rows.append(["sum_cattle_stolen", df["cattle_stolen"].sum()])
if "people_displaced" in df.columns:
    summary_rows.append(["sum_people_displaced", df["people_displaced"].sum()])
if "people_abducted" in df.columns:
    summary_rows.append(["sum_people_abducted", df["people_abducted"].sum()])
if "media_house" in df.columns:
    summary_rows.append(["unique_media_houses", df["media_house"].nunique(dropna=True)])
if "weapon_type" in df.columns:
    summary_rows.append(["unique_weapon_types", df["weapon_type"].nunique(dropna=True)])

executive_summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
executive_summary.to_csv(os.path.join(OUTPUT_DIR, "executive_summary.csv"), index=False)

print("\n==============================")
print("EXECUTIVE SUMMARY")
print("==============================")
print(executive_summary)

# ============================================================
# 21. SAVE CLEANED DATA
# ============================================================
df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_dataset.csv"), index=False)

print("\n==============================")
print("ANALYSIS COMPLETE")
print("==============================")
print(f"Outputs saved in: {OUTPUT_DIR}")
print("Key outputs include:")
print("- cleaned_dataset.csv")
print("- executive_summary.csv")
print("- data_quality_summary.csv")
print("- descriptive_statistics_numeric.csv")
print("- incidents_by_year.csv / .png")
print("- incidents_by_month.csv / .png")
print("- top_places_by_frequency.csv / .png")
print("- deaths_summary.csv")
print("- injuries_summary.csv")
print("- media_house_frequency.csv / .png / wordcloud")
print("- weapon_type_frequency.csv / .png")
print("- other_crimes_frequency.csv / .png / wordcloud")
print("- numeric_correlation_matrix.csv / .png")