from TP1 import scrape_financial_ratios,get_average_return,download_all_historical_data
import json
from datetime import date
import pandas as pd
from TP3 import apply_labeling_to_folder,preprocess_data,train_and_evaluate_models,train_random_forest_model,predict_today
from TP2 import preprocess_financial_clustering,do_kmeans_clustering,do_hierarchical_clustering,prepare_returns_data,preprocess_risk_clustering
from TP4 import predict as predict4
from TP5 import predictfromfile as predict5
from TP6 import update_news,get_three_most_recent_news
from TP8 import get_daily_global_sentiment
def main():
    folder_path = "data/Companies_Historical_Data/"
    R=open("report.html", "w")

    with open('data/companies.json', 'r') as file:
        data = json.load(file)
    ratio_keys = [
        "forwardPE", "beta", "priceToBook", "priceToSales", "dividendYield",
        "trailingEps", "debtToEquity", "currentRatio", "quickRatio",
        "returnOnEquity", "returnOnAssets", "operatingMargins", "profitMargins"
    ]
    ratios=scrape_financial_ratios(data,ratio_keys)
    file_path = "data/ratios_financiers.csv"
    returns_folder = "data/Companies_Historical_Data/"
    data_financial, names_financial = preprocess_financial_clustering(file_path)
    fincluster=do_kmeans_clustering(data_financial, names_financial, n_clusters=5,show_graph=False)

    labeled_data = apply_labeling_to_folder(folder_path )
    print("Prétraitement et extraction des caractéristiques...")
    X, y = preprocess_data(labeled_data)
    model, scaler= train_random_forest_model(X, y)

    from datetime import date

    R.write("""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Rapport Financier</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f9f9f9; padding: 20px; color: #333; }
            h1 { text-align: center; color: #2c3e50; }
            .company-card { background: #fff; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); margin: 20px 0; padding: 20px; }
            h2 { color: #2980b9; }
            h3 { margin-top: 20px; color: #34495e; }
            ul { line-height: 1.6; }
            li { margin-bottom: 5px; }
            .green { color: green; font-weight: bold; }
            .red { color: red; font-weight: bold; }
            .yellow { color: goldenrod; font-weight: bold; }
            .news-title { font-weight: bold; color: #2c3e50; }
            .news-description { margin: 5px 0 10px 0; }
        </style>
    </head>
    <body>
    """)

    R.write(f"<h1>Rapport du {date.today().strftime('%d/%m/%Y')}</h1>")

    for company, row in ratios.iterrows():
        R.write('<div class="company-card">')
        R.write(f"<h2>Entreprise : {company}</h2>")
        
        R.write("<h3>Ratios Importants :</h3><ul>")
        for ratio in ratios.columns:
            value = row[ratio]
            if pd.notnull(value): 
                R.write(f"<li>{ratio} : {value:.2f}</li>")
        R.write("</ul>")

        avg_return = get_average_return(data[company])
        R.write(f"<h3>Rendement journalier moyen : {avg_return:.2%}</h3>")

        if company in fincluster["Name"].values:
            cluster_id = fincluster.loc[fincluster["Name"] == company, "Cluster"].values[0]
            same_cluster_companies = fincluster[fincluster["Cluster"] == cluster_id]["Name"].tolist()
            same_cluster_companies = [name for name in same_cluster_companies if name != company]
            if same_cluster_companies:
                R.write(f"<h3>Entreprises semblables (profils financiers) :</h3><p>{', '.join(same_cluster_companies)}</p>")
            else:
                R.write("<h3>Entreprises semblables (profils financiers) :</h3><p>Aucune</p>")
        else:
            R.write("<h3>Entreprises semblables (profils financiers) :</h3><p>Aucune</p>")

        _, action = predict_today(f"data/Companies_Historical_Data/{company}.csv")
        R.write("<h3>Conseil d’investissement :</h3>")
        if action == "HOLD":
            R.write('<p class="yellow">Hold</p>')
        elif action == "SELL":
            R.write('<p class="red">Vente</p>')
        else:
            R.write('<p class="green">Achat</p>')
        prediction5=float(predict5(company))
        prediction4 = predict4(company)
        price_prediction = (prediction5+float(prediction4['XGBoost']) + float(prediction4['RandomForest']) + float(prediction4['KNN'])) / 4
        R.write(f"<h3>Prédiction du prix de demain :</h3><p>{price_prediction:.2f} €</p>")

        news = get_three_most_recent_news(company)
        R.write("<h3>Actualités :</h3><ul>")
        for newss in news:
            R.write(f"""
                <li>
                    <div class="news-title">{newss['title']}</div>
                    <div class="news-description">{newss['description']}</div>
                    <small>{newss['publishedAt']}</small>
                </li>
            """)
        R.write("</ul>")

        sentiment = get_daily_global_sentiment(company, f"data/news/JSONS/{company.replace(" ","_")}_news.json", "ProsusAI_finetuned/model.safetensors", method="average")
        R.write(f"<h3>Sentiment sur l’actualité du jour :</h3><p>{sentiment}</p>")

        R.write("</div>")  # End of company card

    R.write("</body></html>")
