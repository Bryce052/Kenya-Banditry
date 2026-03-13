import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -----------------------------
# 1. Load the dataset
# -----------------------------
file_name = "newspaper_name.csv"
df = pd.read_csv(file_name)

# -----------------------------
# 2. Clean the Newspaper_Name column
# -----------------------------
df["Newspaper_Name"] = df["Newspaper_Name"].astype(str).str.strip()
df = df[df["Newspaper_Name"].notna()]
df = df[df["Newspaper_Name"] != ""]
df = df[df["Newspaper_Name"].str.lower() != "nan"]

# -----------------------------
# 3. Frequency counts
# -----------------------------
newspaper_counts = df["Newspaper_Name"].value_counts()

print("\nTop 20 Newspapers by Frequency:\n")
print(newspaper_counts.head(20))

print("\nDescriptive Statistics of Newspaper Frequency:\n")
print(newspaper_counts.describe())

# -----------------------------
# 4. Bar Plot - Top 10 Newspapers
# -----------------------------
plt.figure(figsize=(12, 6))
newspaper_counts.head(10).plot(kind="bar")
plt.title("Top 10 Newspapers in Dataset")
plt.xlabel("Newspaper Name")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top_10_newspapers.png", dpi=300)
plt.show()

# -----------------------------
# 5. Histogram - Distribution of Newspaper Frequency
# -----------------------------
plt.figure(figsize=(10, 6))
newspaper_counts.plot(kind="hist", bins=20)
plt.title("Distribution of Newspaper Coverage Frequency")
plt.xlabel("Frequency")
plt.ylabel("Number of Newspapers")
plt.tight_layout()
plt.savefig("newspaper_frequency_distribution.png", dpi=300)
plt.show()

# -----------------------------
# 6. Horizontal Bar Plot - Top 15 Newspapers
# -----------------------------
plt.figure(figsize=(12, 8))
newspaper_counts.head(15).sort_values().plot(kind="barh")
plt.title("Top 15 Newspapers by Frequency")
plt.xlabel("Frequency")
plt.ylabel("Newspaper Name")
plt.tight_layout()
plt.savefig("top_15_newspapers_horizontal.png", dpi=300)
plt.show()

# -----------------------------
# 7. Pie Chart - Top 10 Newspapers
# -----------------------------
plt.figure(figsize=(8, 8))
newspaper_counts.head(10).plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Share of Top 10 Newspapers")
plt.ylabel("")
plt.tight_layout()
plt.savefig("top_10_newspapers_pie.png", dpi=300)
plt.show()

# -----------------------------
# 8. Word Cloud
# -----------------------------
text = " ".join(df["Newspaper_Name"].dropna())

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(text)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Newspaper Names")
plt.tight_layout()
plt.savefig("newspaper_wordcloud.png", dpi=300)
plt.show()

# -----------------------------
# 9. Save frequency table to CSV
# -----------------------------
newspaper_counts.reset_index().rename(
    columns={"index": "Newspaper_Name", "Newspaper_Name": "Frequency"}
).to_csv("newspaper_frequency_table.csv", index=False)

print("\nAnalysis complete.")
print("Saved outputs:")
print("- top_10_newspapers.png")
print("- newspaper_frequency_distribution.png")
print("- top_15_newspapers_horizontal.png")
print("- top_10_newspapers_pie.png")
print("- newspaper_wordcloud.png")
print("- newspaper_frequency_table.csv")