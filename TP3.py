import pandas as pd
import glob
import ta  # Technical Analysis library
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,f1_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands
import shap
import matplotlib.pyplot as plt
import warnings
import joblib
warnings.filterwarnings("ignore")
import argparse

# === Création des labels en 3 classes : sell (0), hold (1), buy (2) ===
def create_labels(df):
    df = df[['Close']].copy()
    df['Close Horizon'] = df['Close'].shift(-20)  # Cours dans 20 jours
    df['horizon return'] = (df['Close Horizon'] - df['Close']) / df['Close']
    df['label'] = df['horizon return'].apply(
        lambda x: 2 if x > 0.05 else (0 if x < -0.05 else 1)
    )
    return df

# === Application de l'étiquetage à tous les fichiers CSV d'un dossier ===
def apply_labeling_to_folder(folder_path):
    print(folder_path)
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    labeled_data = {}
    for file in all_files:
        try:
            df = pd.read_csv(file)
            labels = create_labels(df)
            df = df.iloc[:-20, :]  # Supprimer les dernières lignes (NaN dû au shift)
            labels = labels.iloc[:-20, :]
            df['label'] = labels['label']
            stock_name = os.path.basename(file).split(".")[0]
            labeled_data[stock_name] = df
        except Exception as e:
            print(f"Erreur avec le fichier {file} : {e}")
    return labeled_data

# === Ajout des indicateurs techniques à un DataFrame ===
def add_technical_indicators(df):
    df['SMA 20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA 20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI 14'] = RSIIndicator(df['Close'], window=14).rsi()

    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD Signal'] = macd.macd_signal()

    boll = BollingerBands(df['Close'])
    df['Bollinger High'] = boll.bollinger_hband()
    df['Bollinger Low'] = boll.bollinger_lband()

    df['Rolling Volatility 20'] = df['Close'].rolling(window=20).std()
    df['ROC 10'] = ROCIndicator(df['Close'], window=10).roc()

    return df

# === Préparation des features (X) et labels (y) à partir des données étiquetées ===
# Add debug statements to inspect the data at each step

def preprocess_data(labeled_data):
    X, y = [], []
    for stock, df in labeled_data.items():
        print(f"Processing stock: {stock}, initial shape: {df.shape}")  # Debug
        df = add_technical_indicators(df)
        print(f"After adding indicators, shape: {df.shape}")  # Debug
        df = df.dropna()
        print(f"After dropping NaN, shape: {df.shape}")  # Debug
        if 'label' in df.columns:
            numeric_df = df.drop(columns=['label']).select_dtypes(include=['number'])
            X.append(numeric_df)
            y.append(df['label'])
        else:
            print(f"Warning: 'label' column missing for stock: {stock}")  # Debug
    if not X or not y:
        print("Error: No valid data found in labeled_data.")  # Debug
    X = pd.concat(X, axis=0)
    y = pd.concat(y, axis=0)
    print(f"Final shapes - X: {X.shape}, y: {y.shape}")  # Debug
    return X, y

# === Fonction générique GridSearchCV ===
def optimize_with_gridsearch(model, param_grid, X_train, y_train, name):
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"\nMeilleurs paramètres pour {name} : {grid.best_params_}")
    return grid.best_estimator_

# === Fonction principale d'entraînement et d'évaluation de tous les modèles ===
def train_and_evaluate_models(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5]
    }
    xgb_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 10],
        "learning_rate": [0.01, 0.1]
    }
    knn_param_grid = {
        "n_neighbors": [3, 5, 7],
        "weights": ['uniform', 'distance']
    }
    svm_param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ['linear', 'rbf']
    }
    logreg_param_grid = {
        "C": [0.1, 1, 10],
        "solver": ['lbfgs'],
        "max_iter": [1000]
    }

    # Random Forest
    print("\n=== Optimisation Random Forest ===")
    rf_best = optimize_with_gridsearch(RandomForestClassifier(), rf_param_grid, X_train, y_train, "Random Forest")
    y_pred_rf = rf_best.predict(X_test)
    print("\n=== Random Forest ===")
    print(classification_report(y_test, y_pred_rf))

    # XGBoost
    print("\n=== Optimisation XGBoost ===")
    xgb_best = optimize_with_gridsearch(XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), xgb_param_grid, X_train, y_train, "XGBoost")
    y_pred_xgb = xgb_best.predict(X_test)
    print("\n=== XGBoost ===")
    print(classification_report(y_test, y_pred_xgb))

    # KNN
    print("\n=== Optimisation KNN ===")
    knn_best = optimize_with_gridsearch(KNeighborsClassifier(), knn_param_grid, X_train, y_train, "KNN")
    y_pred_knn = knn_best.predict(X_test)
    print("\n=== KNN ===")
    print(classification_report(y_test, y_pred_knn))

    # SVM
    print("\n=== Optimisation SVM ===")
    svm_best = optimize_with_gridsearch(SVC(), svm_param_grid, X_train, y_train, "SVM")
    y_pred_svm = svm_best.predict(X_test)
    print("\n=== SVM ===")
    print(classification_report(y_test, y_pred_svm))

    # Logistic Regression
    print("\n=== Optimisation Logistic Regression ===")
    logreg_best = optimize_with_gridsearch(LogisticRegression(), logreg_param_grid, X_train, y_train, "Logistic Regression")
    y_pred_logreg = logreg_best.predict(X_test)
    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, y_pred_logreg))

    # Retourne le modèle le plus interprétable pour SHAP (ici Random Forest)
    return rf_best, X_train, X_test

#as Random Forest shows thee best performance/speed ratio, we decided to use for the final project
def train_random_forest_model(X, y, model_path="models/RF/random_forest_model.pkl", scaler_path="models/RF/scaler.pkl"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')

    print("=== Random Forest ===")
    print(classification_report(y_test, y_pred))
    print(f"F1 Score (weighted): {score:.4f}")

    # Save the model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler

def predict_today(file_path, model_path="models/RF/random_forest_model.pkl", scaler_path="models/RF/scaler.pkl"):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        df = pd.read_csv(file_path)
        df = add_technical_indicators(df)
        df = df.dropna()
        numeric_df = df.select_dtypes(include=['number']).drop(columns=['label'], errors='ignore')
        X_new = scaler.transform(numeric_df)
        predictions = model.predict(X_new)
        df['prediction'] = predictions
        latest_prediction = predictions[-1]
        if latest_prediction == 2:
            action = "BUY"
        elif latest_prediction == 0:
            action = "SELL"
        else:
            action = "HOLD"
        print(f"Trading signal for today ({df.index[-1]}): {action}")
        return df, action
    except Exception as e:
        print(f"Error with the file {file_path} : {e}")
        return None, None

# === Visualisation SHAP pour interprétation du modèle Random Forest ===
def explain_with_shap(model, X_train, X_test, feature_names):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names)

# === Pipeline principal exécutant tout le processus ===
def main():
    folder_path = "data/Companies_Historical_Data/"
    print("Chargement et étiquetage des données...")
    labeled_data = apply_labeling_to_folder(folder_path )
    print("Prétraitement et extraction des caractéristiques...")
    X, y = preprocess_data(labeled_data)
    print("Entraînement des modèles et évaluation avec GridSearchCV...")
    rf_model, X_train, X_test = train_and_evaluate_models(X, y)
    print("Interprétation SHAP du modèle Random Forest...")
    explain_with_shap(rf_model, X_train, X_test, feature_names=X.columns)

# === Lancement du script si exécuté directement ===
if __name__ == "__main__":
    main()
