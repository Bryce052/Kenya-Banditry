import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def extract_details(article_text):
    """Extract detailed information from the article text."""
    def safe_search(pattern, text):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1) if match else None

    details = {
        "year": safe_search(r"\b(20\d{2})\b", article_text),
        "month": safe_search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", article_text),
        "place": safe_search(r"(?:in|at|near|around)\s([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", article_text),
        "attacking_bandits": safe_search(r"\b(\d+)\s+bandits\b", article_text),
        "livestock_stolen": "Yes" if re.search(r"(?:livestock|cattle)\s+(?:was|were)\s+stolen", article_text, re.IGNORECASE) else "No",
        "number_livestock_stolen": safe_search(r"\b(\d+)\s+(?:livestock|cattle)\b", article_text),
        "total_deaths": safe_search(r"\b(\d+)\s+(?:deaths|people\s+dead)\b", article_text),
        "police_deaths": safe_search(r"\b(\d+)\s+police\s+(?:deaths|killed)\b", article_text),
        "locals_deaths": safe_search(r"\b(\d+)\s+locals\s+(?:deaths|killed)\b", article_text),
        "bandit_deaths": safe_search(r"\b(\d+)\s+bandits\s+(?:deaths|killed)\b", article_text),
        "total_injuries": safe_search(r"\b(\d+)\s+injuries\b", article_text),
        "police_injured": safe_search(r"\b(\d+)\s+police\s+injured\b", article_text),
        "locals_injured": safe_search(r"\b(\d+)\s+locals\s+injured\b", article_text),
        "bandits_injured": safe_search(r"\b(\d+)\s+bandits\s+injured\b", article_text),
        "displaced_people": safe_search(r"\b(\d+)\s+people\s+displaced\b", article_text),
        "abducted_people": safe_search(r"\b(\d+)\s+people\s+abducted\b", article_text),
        "other_crimes": safe_search(r"(arson|kidnapping|theft|rape|extortion)", article_text),
        "weapon_type": safe_search(r"(guns|rifles|AK-47s|pistols|weapons)", article_text),
        "specific_weapon": safe_search(r"\b(AK-47|pistol|rifle)\b", article_text),
        "number_weapons": safe_search(r"\b(\d+)\s+(?:weapons|guns)\b", article_text),
        "ammunition": safe_search(r"\b(\d+)\s+(?:rounds|bullets|ammunition)\b", article_text),
        "facilities_attacked": safe_search(r"(police station|school|hospital|market)", article_text),
    }
    return details

def scrape_single_url(url):
    """Scrape a single URL and extract detailed information."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch URL: {url}, Status Code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        article_text = soup.get_text(separator=" ").strip()

        # Extract details from the article text
        details = extract_details(article_text)
        details.update({"url": url})

        return details

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def main():
    url = input("Enter the URL to scrape: ")
    print(f"Scraping URL: {url}")
    article_details = scrape_single_url(url)

    if article_details:
        # Save results to a CSV file
        df = pd.DataFrame([article_details])
        df.to_csv("scraped_article_details.csv", index=False)
        print("Article details saved to scraped_article_details.csv")
    else:
        print("No details extracted.")

if __name__ == "__main__":
    main()
