import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import pytz
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from collections import defaultdict
from matplotlib.lines import Line2D
import glob

# 1.1 Extraction des textes et timestamps
def get_texts_timestamps(news_data):
    texts = []
    timestamps = []
    ny_tz = pytz.timezone("America/New_York")
    for article in news_data:
        timestamp_utc = datetime.fromisoformat(article['timestamp'].replace("Z", "+00:00"))
        timestamp_ny = timestamp_utc.astimezone(ny_tz)
        rounded_timestamp = timestamp_ny.replace(minute=0, second=0, microsecond=0)
        text = article.get("title", "") + " " + article.get("description", "")
        texts.append(text.strip())
        timestamps.append(rounded_timestamp)
    return texts, timestamps

# 1.2 Analyse de sentiments
def get_sentiments(model_path, texts):
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    sentiments = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        sentiments.append(pred)
    return sentiments

# 1.3 Alignement des timestamps
def align_timestamps(timestamps):
    aligned = []
    market_open = datetime.strptime("09:30", "%H:%M").time()
    market_close = datetime.strptime("15:00", "%H:%M").time()

    for ts in timestamps:
        time = ts.time()
        if market_open <= time < market_close:
            aligned.append(ts.replace(minute=0, second=0, microsecond=0))
        elif market_close <= time < datetime.strptime("23:59", "%H:%M").time():
            aligned.append(ts.replace(hour=15, minute=0, second=0, microsecond=0))
        else:  # early morning hours
            aligned.append((ts - timedelta(days=1)).replace(hour=15, minute=0, second=0, microsecond=0))
    return aligned

# 1.4 Visualisation
def plot_comparison(df, sentiments_a, sentiments_b, timestamps, title_a, title_b):
    aligned_ts = align_timestamps(timestamps)
    
    def group_by_time(ts_list, sentiments_list):
        grouped = defaultdict(list)
        for t, s in zip(ts_list, sentiments_list):
            grouped[t].append(s)
        return grouped

    grouped_a = group_by_time(aligned_ts, sentiments_a)
    grouped_b = group_by_time(aligned_ts, sentiments_b)

    def plot_sub(ax, grouped, title):
        df.set_index("Datetime", inplace=True)
        ax.plot(df.index, df["Close"], label="Price", color="black")
        colors = {0: "red", 1: "orange", 2: "green"}
        offset = 0.5

        for t, s_list in grouped.items():
            for i, s in enumerate(s_list):
                if t in df.index:
                    price = df.loc[t]["Close"]
                    ax.scatter(t, price + i * offset, color=colors[s], s=60)
        ax.set_title(title)
        ax.set_ylabel("Price")
        ax.grid(True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    plot_sub(ax1, grouped_a, title_a)
    plot_sub(ax2, grouped_b, title_b)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Neutral', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='black', lw=2, label='Price')
    ]
    ax2.legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()

# 1.5 Intégration
def run_analysis(company, json_path, model_path_a, model_path_b):
    # Télécharger les données horaires
    ticker = yf.Ticker(company)
    df = ticker.history(start="2025-01-01", interval="60m").reset_index()

    # Charger les news
    with open(json_path, 'r') as f:
        news_data = json.load(f)
    
    texts, timestamps = get_texts_timestamps(news_data)

    sentiments_a = get_sentiments(model_path_a, texts)
    sentiments_b = get_sentiments(model_path_b, texts)

    plot_comparison(df, sentiments_a, sentiments_b, timestamps, "Model A", "Model B")

def count_news_per_company(json_paths):
    summary = {}
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
        total_news = sum(len(v) for v in data.values())
        company = os.path.basename(path).replace("_news.json", "")
        summary[company] = total_news
    return {k: v for k, v in sorted(summary.items(), key=lambda x: x[1], reverse=True)}


json_files = glob.glob("./*_news.json")
news_counts = count_news_per_company(json_files)

for company, count in news_counts.items():
    print(f"{company}: {count} news")

run_analysis(
    company="AAPL",  # ou "AMZN", etc.
    json_path="./Apple_news.json",
    model_path_a="./finbert_base",          # FinBERT original
    model_path_b="./finbert_finetuned"      # modèle fine-tuné par ton camarade
)
