from scraping import update_news

# Clé API
api_key = "votre clé_api_ici" # Remplacez par votre clé API

# Liste d'entreprises à mettre à jour automatiquement
companies = ["Apple", "Tesla", "Microsoft", "Amazon", "Google"]

for company in companies:
    print(f"Mise à jour de : {company}")
    update_news(company, api_key)