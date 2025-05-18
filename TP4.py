import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# ----- Étapes de prétraitement -----
def load_close_prices(file_path):
    df = pd.read_csv(file_path)
    return df[['Close']]

def scale_and_split(data, split_ratio=0.8):
    scaler = MinMaxScaler()
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler

def create_target_features(df_scaled, n_days=30):
    x, y = [], []
    for i in range(n_days, len(df_scaled)):
        x.append(df_scaled[i - n_days:i, 0])
        y.append(df_scaled[i, 0])
    return np.array(x), np.array(y)

def prepare_dataset(file_path, n_days=30, split_ratio=0.8):
    df = load_close_prices(file_path)
    train_scaled, test_scaled, scaler = scale_and_split(df, split_ratio)
    x_train, y_train = create_target_features(train_scaled, n_days)
    x_test, y_test = create_target_features(test_scaled, n_days)
    return x_train, y_train, x_test, y_test, scaler

# ----- Entraînement et évaluation des modèles -----
def train_models(x_train, y_train):
    models = {
        'XGBoost': XGBRegressor(n_estimators=100, verbosity=0),
        'RandomForest': RandomForestRegressor(n_estimators=100),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    for name, model in models.items():
        model.fit(x_train, y_train)
    return models

def evaluate_models(models, x_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Compatible avec toutes versions sklearn
        results[name] = {'MAE': mae, 'RMSE': rmse}
    return results

# ----- Visualisation -----
def plot_predictions(y_test, y_pred, title, output_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(y_test, label='Réel', linewidth=2)
    plt.plot(y_pred, label='Prévu', linewidth=2)
    plt.title(title)
    plt.xlabel("Temps")
    plt.ylabel("Valeur normalisée")
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()

# ----- Pipeline principal -----
def run_preprocessing_pipeline(data_folder="data/Companies_Historical_Data",
                               output_folder="models/TP4",
                               n_days=30, split_ratio=0.8):
    os.makedirs(output_folder, exist_ok=True)
    file_list = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

    # Dictionnaire pour stocker les scores de chaque modèle
    all_scores = {}
    for filename in file_list:
        name = os.path.splitext(filename)[0]
        file_path = os.path.join(data_folder, filename)

        # Prétraitement
        x_train, y_train, x_test, y_test, scaler = prepare_dataset(file_path, n_days, split_ratio)

        # Entraînement
        models = train_models(x_train, y_train)

        # Évaluation
        results = evaluate_models(models, x_test, y_test)
        print(f"\n{name} - Résultats :")
        for model_name, metrics in results.items():
            print(f"{model_name} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            # Stocker les scores pour la moyenne
            if model_name not in all_scores:
                all_scores[model_name] = {'MAE': [], 'RMSE': []}
            all_scores[model_name]['MAE'].append(metrics['MAE'])
            all_scores[model_name]['RMSE'].append(metrics['RMSE'])

        # Visualisation
        for model_name, model in models.items():
            y_pred = model.predict(x_test)
            fig_path = os.path.join(output_folder, f"{name}_{model_name}_prediction.png")
            plot_predictions(y_test, y_pred, title=f"{name} - {model_name}", output_path=fig_path)

        # Sauvegarde des données
        np.save(os.path.join(output_folder, f"{name}_x_train.npy"), x_train)
        np.save(os.path.join(output_folder, f"{name}_y_train.npy"), y_train)
        np.save(os.path.join(output_folder, f"{name}_x_test.npy"), x_test)
        np.save(os.path.join(output_folder, f"{name}_y_test.npy"), y_test)
        joblib.dump(scaler, os.path.join(output_folder, f"{name}_scaler.pkl"))

    # Calcul et affichage des moyennes
    print("\nMoyenne des MAE et RMSE pour chaque modèle :")
    for model_name, scores in all_scores.items():
        mean_mae = np.mean(scores['MAE'])
        mean_rmse = np.mean(scores['RMSE'])
        print(f"{model_name} - MAE moyen: {mean_mae:.4f}, RMSE moyen: {mean_rmse:.4f}")

def predict(company_name, model_folder="models/TP4"):
    import os

    # Load saved test data and scaler
    x_test_path = os.path.join(model_folder, f"{company_name}_x_test.npy")
    scaler_path = os.path.join(model_folder, f"{company_name}_scaler.pkl")

    if not os.path.exists(x_test_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing files for company: {company_name}")

    x_test = np.load(x_test_path)
    scaler = joblib.load(scaler_path)

    # Prepare the last sample for prediction
    last_sample = x_test[-1].reshape(1, -1)

    # Load trained models (retrain them from saved training data)
    x_train = np.load(os.path.join(model_folder, f"{company_name}_x_train.npy"))
    y_train = np.load(os.path.join(model_folder, f"{company_name}_y_train.npy"))
    models = train_models(x_train, y_train)

    # Make predictions and inverse-transform them
    predictions = {}
    for name, model in models.items():
        pred_scaled = model.predict(last_sample)[0]
        # Inverse transform expects shape (n_samples, 1)
        pred_unscaled = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1))[0][0]
        predictions[name] = pred_unscaled

    return predictions

# ----- Exécution -----
if __name__ == "__main__":
    run_preprocessing_pipeline()