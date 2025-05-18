import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Liste des entreprises
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

# Ratios financiers à collecter
ratio_keys = [
    "forwardPE", "beta", "priceToBook", "priceToSales", "dividendYield",
    "trailingEps", "debtToEquity", "currentRatio", "quickRatio",
    "returnOnEquity", "returnOnAssets", "operatingMargins", "profitMargins"
]

def scrape_financial_ratios(companies, ratio_keys, output_file="data/ratios_financiers.csv"):
    print("Scraping des ratios financiers...")
    ratios = {key: [] for key in ratio_keys}
    failed = []

    for name, symbol in companies.items():
        try:
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            for key in ratio_keys:
                ratios[key].append(info.get(key, None))
        except Exception as e:
            print(f"Erreur pour {name} ({symbol}) : {e}")
            for key in ratio_keys:
                ratios[key].append(None)
            failed.append(name)

    df = pd.DataFrame(ratios, index=companies.keys())
    df.to_csv(output_file,index_label="Name")
    return(df)
    print(f"Ratios sauvegardés dans {output_file}")
    if failed:
        print(f"Entreprises en échec : {failed}")

def download_all_historical_data(companies, output_dir="data/Companies_Historical_Data", years=5):
    """
    Télécharge les prix de clôture sur N années et calcule le rendement journalier.
    Enregistre chaque entreprise dans un fichier CSV séparé.

    Args:
        companies (dict): Dictionnaire {nom_entreprise: ticker_yahoo}
        output_dir (str): Dossier de sauvegarde
        years (int): Nombre d'années à récupérer
    """
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    os.makedirs(output_dir, exist_ok=True)

    for company, symbol in companies.items():
        try:
            df = yf.download(symbol, start=start_date, end=end_date)

            # Si MultiIndex (certains marchés), on garde le premier niveau
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[['Close']]
            df['Next Day Close'] = df['Close'].shift(-1)
            df['Rendement'] = (df['Next Day Close'] - df['Close']) / df['Close']
            df.dropna(inplace=True)

            df.to_csv(os.path.join(output_dir, f"{company}.csv"))
            print(f"{company} : sauvegardé.")
        except Exception as e:
            print(f"{company} : erreur — {e}")
def get_average_return(ticker,  years=5):
    """
    Télécharge les prix de clôture sur N années pour un ticker et calcule le rendement journalier moyen.

    Args:
        ticker (str): Le ticker de l'entreprise.
        years (int): Nombre d'années à récupérer.

    Returns:
        float: Le rendement journalier moyen.
    """
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    try:
        df = yf.download(ticker, start=start_date, end=end_date)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Close']]
        df['Next Day Close'] = df['Close'].shift(-1)
        df['Rendement'] = (df['Next Day Close'] - df['Close']) / df['Close']
        df.dropna(inplace=True)

        return df['Rendement'].mean()

    except Exception as e:
        print(f"{ticker} : erreur — {e}")
        return None
def run_full_pipeline():
    scrape_financial_ratios(companies, ratio_keys)
    download_all_historical_data(companies)

# Lancer tout
if __name__ == "__main__":
    run_full_pipeline()