import requests
import json
from datetime import datetime, timedelta
import os

# Charger les news existantes depuis un fichier local
def load_existing_news(company_name):
    file_path = f"data/news/JSONS/{company_name}_news.json"
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
def get_three_most_recent_news(company_name):
    file_path = f"data/news/JSONS/{company_name}_news.json"
    if not os.path.exists(file_path):
        print(f"Aucun fichier trouvé pour {company_name}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    all_articles = []
    for articles in news_data.values():  # Loop directly through lists of articles
        for article in articles:
            published_at = article.get("publishedAt")
            if published_at:
                try:
                    dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))  # Convert to datetime
                except ValueError:
                    dt = datetime.min
                all_articles.append((dt, article))

    # Sort articles by datetime (most recent first)
    all_articles.sort(key=lambda x: x[0], reverse=True)

    # Return the top 3 articles only
    return [article for _, article in all_articles[:3]]

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
        "JP_Morgan": "JPM",
        "Goldman_Sachs": "GS",
        "Visa": "V",
        "Johnson_& _Johnson": "JNJ",
        "Pfizer": "PFE",
        "ExxonMobil": "XOM",
        "ASML": "ASML.AS",
        "SAP": "SAP.DE",
        "Siemens": "SIE.DE",
        "Louis_Vuitton_(LVMH)": "MC.PA",
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
        "Reliance_Industries": "RELIANCE.NS",
        "Tata_Consultancy_Services": "TCS.NS"
    }

    for company in companies:
        print(f"Mise à jour de : {company}")
        update_news(company, api_key)