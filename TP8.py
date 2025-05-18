import json
import os
from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import pytz
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from collections import defaultdict
from matplotlib.lines import Line2D
import glob
from bisect import bisect_left

companies = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN", "Alphabet": "GOOGL", "Meta": "META",
    "Tesla": "TSLA", "NVIDIA": "NVDA", "Samsung": "005930.KS", "Tencent": "TCEHY", "Alibaba": "BABA",
    "IBM": "IBM", "Intel": "INTC", "Oracle": "ORCL", "Sony": "SONY", "Adobe": "ADBE",
    "Netflix": "NFLX", "AMD": "AMD", "Qualcomm": "QCOM", "Cisco": "CSCO", "JP Morgan": "JPM",
    "Goldman Sachs": "GS", "Visa": "V", "Johnson & Johnson": "JNJ", "Pfizer": "PFE",
    "ExxonMobil": "XOM", "ASML": "ASML.AS", "SAP": "SAP.DE", "Siemens": "SIE.DE",
    "Louis Vuitton (LVMH)": "MC.PA", "TotalEnergies": "TTE.PA", "Shell": "SHEL.L",
    "Baidu": "BIDU", "JD.com": "JD", "BYD": "BYDDY", "ICBC": "1398.HK", "Toyota": "TM",
    "SoftBank": "9984.T", "Nintendo": "NTDOY", "Hyundai": "HYMTF", "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS"
}

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def clean_text(text):
    return text.strip().replace("\n", " ").replace("\r", " ")

def convert_utc_to_ny(timestamp_str):
    utc_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    ny_tz = pytz.timezone("America/New_York")
    ny_dt = utc_dt.astimezone(ny_tz)
    return ny_dt.replace(minute=0, second=0, microsecond=0)

def get_texts_timestamps(news_data):
    texts = []
    timestamps = []
    for day_articles in news_data.values():
        for article in day_articles:
            ts = convert_utc_to_ny(article['publishedAt'])
            text = clean_text(article.get("title", "") + " " + article.get("description", ""))
            texts.append(text)
            timestamps.append(ts)
    return texts, timestamps

def get_sentiments(model_path, texts):
    log(f"Chargement du modèle depuis {model_path}")
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    sentiments = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        sentiments.append(pred)
    return sentiments

def align_timestamps(timestamps):
    from pandas.tseries.holiday import USFederalHolidayCalendar

    aligned = []
    ny_tz = pytz.timezone("America/New_York")
    calendar = USFederalHolidayCalendar()
    holidays = set(h.date() for h in calendar.holidays(start="2025-01-01", end="2025-12-31"))

    for ts in timestamps:
        ts = ts.astimezone(ny_tz)
        local_date = ts.date()
        local_time = ts.time()
        weekday = ts.weekday()  # 0=lundi, 6=dimanche

        # Cas 1 : weekend → reculer au vendredi précédent
        if weekday == 5:  # samedi
            target = ts - timedelta(days=1)
        elif weekday == 6:  # dimanche
            target = ts - timedelta(days=2)
        # Cas 2 : jour férié
        elif local_date in holidays:
            target = ts - timedelta(days=1)
            while target.date() in holidays or target.weekday() >= 5:
                target -= timedelta(days=1)
        # Cas 3 : jour de marché
        else:
            if datetime.strptime("09:30", "%H:%M").time() <= local_time < datetime.strptime("15:00", "%H:%M").time():
                aligned.append(ts.replace(minute=0, second=0, microsecond=0))
                continue
            elif local_time >= datetime.strptime("15:00", "%H:%M").time():
                aligned.append(ts.replace(hour=15, minute=0, second=0, microsecond=0))
                continue
            else:
                target = ts - timedelta(days=1)

        # S'assurer que le timestamp aligné est localisé New York
        aligned_dt = datetime.combine(target.date(), time(15, 0))
        aligned.append(ny_tz.localize(aligned_dt))

    return aligned


def plot_comparison(df, sentiments_a, sentiments_b, timestamps, title_a, title_b):
    aligned_ts = align_timestamps(timestamps)

    def group_by_time(ts_list, sentiments_list):
        grouped = defaultdict(list)
        for t, s in zip(ts_list, sentiments_list):
            grouped[t].append(s)
        return grouped

    grouped_a = group_by_time(aligned_ts, sentiments_a)
    grouped_b = group_by_time(aligned_ts, sentiments_b)

    def plot_sub(df, ax, grouped, title):
        df = df.set_index("Datetime" if "Datetime" in df.columns else df.columns[0])
        index_list = df.index.to_list()
        ax.plot(df.index, df["Close"], label="Price", color="black")
        colors = {0: "red", 1: "orange", 2: "green"}
        offset = 0.5

        for t, s_list in grouped.items():
            pos = bisect_left(index_list, t)
            if pos == len(index_list):
                continue
            nearest = index_list[pos] if abs(index_list[pos] - t) <= timedelta(minutes=90) else None
            if nearest:
                price = df.loc[nearest]["Close"]
                for i, s in enumerate(s_list):
                    ax.scatter(nearest, price + i * offset, color=colors[s], s=60)

        ax.set_title(title)
        ax.set_ylabel("Price")
        ax.grid(True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    plot_sub(df, ax1, grouped_a, title_a)
    plot_sub(df, ax2, grouped_b, title_b)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Neutral', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='black', lw=2, label='Price')
    ]
    ax2.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()

def run_analysis(company, json_path, model_path_a, model_path_b):
    log(f"Téléchargement des prix pour {company}")
    ticker_symbol = companies.get(company, company)
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start="2025-01-01", end="2025-04-15", interval="60m")
    df = df.reset_index() if 'Datetime' not in df.columns else df

    log(f"Chargement des news depuis {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        news_data = json.load(f)

    texts, timestamps = get_texts_timestamps(news_data)
    sentiments_a = get_sentiments(model_path_a, texts)
    sentiments_b = get_sentiments(model_path_b, texts)
    plot_comparison(df, sentiments_a, sentiments_b, timestamps, "Model ProsusAI pour " + company, "Model Finbert pour " + company)

def count_news_per_company(json_dir):
    summary = {}
    for path in glob.glob(os.path.join(json_dir, "*_news.json")):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total = sum(len(v) for v in data.values())
        company = os.path.basename(path).replace("_news.json", "")
        summary[company] = (total, path)
    return {k: v for k, v in sorted(summary.items(), key=lambda x: x[1][0], reverse=True)}

def get_daily_global_sentiment(company, news_json_path, model_path, method="average"):
    """
    Retourne le sentiment global des news du jour pour une entreprise donnée.

    Args:
        company (str): Nom de l'entreprise.
        news_json_path (str): Chemin vers le fichier JSON contenant les news.
        model_path (str): Chemin du modèle ProsusAI finetuned.
        method (str): Méthode d’agrégation ("average", "majority", "distribution").

    Returns:
        str ou dict: Sentiment global du jour (ou distribution).
    """
    import json
    from datetime import datetime
    import pytz
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    import numpy as np
    from collections import Counter

    # Charger les news
    with open(news_json_path, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    # Date du jour en timezone UTC pour matcher les formats
    today = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")

    # Extraire les articles du jour
    articles = news_data.get(today, [])
    if not articles:
        return "Aucune news pour aujourd’hui."

    # Préparer les textes
    texts = [
        (article.get("title", "") + " " + article.get("description", "")).strip().replace("\n", " ").replace("\r", " ")
        for article in articles
    ]

    # Charger modèle/tokenizer
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Prédiction sentiments
    sentiments = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        sentiments.append(pred)

    # Agrégation
    if method == "average":
        avg = np.mean(sentiments)
        if avg < 0.75:
            return "Négatif"
        elif avg < 1.5:
            return "Neutre"
        else:
            return "Positif"
    elif method == "majority":
        count = Counter(sentiments)
        most_common = count.most_common(1)[0][0]
        mapping = {0: "Négatif", 1: "Neutre", 2: "Positif"}
        return mapping.get(most_common, "Indéterminé")
    elif method == "distribution":
        count = Counter(sentiments)
        total = len(sentiments)
        return {
            "Négatif": count[0]/total,
            "Neutre": count[1]/total,
            "Positif": count[2]/total
        }
    else:
        raise ValueError("Méthode d’agrégation non reconnue.")


if __name__ == "__main__":
    news_counts = count_news_per_company("JSONS")
    print("Entreprises avec le plus de news :")
    for company, (count, _) in news_counts.items():
        print(f"{company}: {count} news")

    top_2_companies = list(news_counts.items())[:2]
    for company, (_, path) in top_2_companies:
        log(f"Analyse pour {company}")
        run_analysis(
            company=company,
            json_path=path,
            model_path_a="./ProsusAI_finetuned",
            model_path_b="./finbert_finetuned"
        )
