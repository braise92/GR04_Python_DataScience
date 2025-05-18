import requests
import json
from datetime import datetime, timedelta
import os

# Charger les news existantes depuis un fichier local
def load_existing_news(company_name):
    file_path = os.path.join("News", f"{company_name.lower().replace(' ', '_')}_news.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Récupérer les actualités via l'API pour une entreprise donnée
def get_news_by_date(company_name, api_key):
    url = 'https://newsapi.org/v2/everything'
    last_day = datetime.today().strftime('%Y-%m-%d')
    first_day = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')

    params = {
        "sources": 'financial-post,the-wall-street-journal,bloomberg,the-washington-post,australian-financial-review,bbc-news,cnn',
        "q": company_name,
        "apiKey": api_key,
        "language": "en",
        "pageSize": 100,
        "from": first_day,
        "to": last_day,
    }

    response = requests.get(url, params=params)
    news_dict = {}

    if response.status_code == 200:
        articles = response.json().get("articles", [])
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            published_at = article.get("publishedAt", "")
            source_name = article.get("source", {}).get("name", "")

            if company_name.lower() in (title or "").lower() or company_name.lower() in (description or "").lower():
                date_str = published_at.split("T")[0]
                if date_str not in news_dict:
                    news_dict[date_str] = []
                news_dict[date_str].append({
                    "title": title,
                    "description": description,
                    "source": source_name,
                    "published_at": published_at
                })
    else:
        print(f"Erreur {response.status_code} : {response.text}")

    return news_dict

# Mettre à jour les news d'une entreprise et les sauvegarder localement
def update_news(company_name, api_key):
    old_news = load_existing_news(company_name)
    new_news = get_news_by_date(company_name, api_key)

    for date, articles in new_news.items():
        if date not in old_news:
            old_news[date] = articles
        else:
            existing_titles = {a["title"] for a in old_news[date]}
            for article in articles:
                if article["title"] not in existing_titles:
                    old_news[date].append(article)

    os.makedirs("News", exist_ok=True)
    file_path = os.path.join("News", f"{company_name.lower().replace(' ', '_')}_news.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(old_news, f, indent=2)

    print(f"Fichier mis à jour : {file_path}")

# Programme principal
if __name__ == '__main__':
    # Clé API à renseigner
    api_key = "Clé API personnelle"  # Remplacer par votre clé API

    # Liste complète d'entreprises à mettre à jour automatiquement
    companies = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Alphabet": "GOOGL",
        "Meta": "META",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "Samsung": "005930.KS",
        "Tencent": "TCEHY",
        "Alibaba": "BABA",
        "IBM": "IBM",
        "Intel": "INTC",
        "Oracle": "ORCL",
        "Sony": "SONY",
        "Adobe": "ADBE",
        "Netflix": "NFLX",
        "AMD": "AMD",
        "Qualcomm": "QCOM",
        "Cisco": "CSCO",
        "JP Morgan": "JPM",
        "Goldman Sachs": "GS",
        "Visa": "V",
        "Johnson & Johnson": "JNJ",
        "Pfizer": "PFE",
        "ExxonMobil": "XOM",
        "ASML": "ASML.AS",
        "SAP": "SAP.DE",
        "Siemens": "SIE.DE",
        "Louis Vuitton (LVMH)": "MC.PA",
        "TotalEnergies": "TTE.PA",
        "Shell": "SHEL.L",
        "Baidu": "BIDU",
        "JD.com": "JD",
        "BYD": "BYDDY",
        "ICBC": "1398.HK",
        "Toyota": "TM",
        "SoftBank": "9984.T",
        "Nintendo": "NTDOY",
        "Hyundai": "HYMTF",
        "Reliance Industries": "RELIANCE.NS",
        "Tata Consultancy Services": "TCS.NS"
    }

    for company in companies:
        print(f"Mise à jour de : {company}")
        update_news(company, api_key)