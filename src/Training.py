import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.stats import boxcox


df = pd.read_csv("../data/processed/processed.csv")

X = df.drop(["gross"], axis=1)
y = df["gross"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

train = pd.concat([X_train, y_train], axis=1)
train.to_csv("train.csv", index=False)

test = pd.concat([X_test, y_test], axis=1)
test.to_csv("test.csv", index=False)

pipeline_lasso_pca = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures()),
        ("pca", PCA()),
        ("lasso", Lasso()),
    ]
)

parametros_grid_pca = {
    "poly__degree": [2, 3, 4, 5],
    "pca__n_components": [None, 2, 5, 10],
    "lasso__alpha": [0.1, 1.0, 10.0],
}

grid_search_pca = GridSearchCV(
    pipeline_lasso_pca,
    parametros_grid_pca,
    scoring="neg_mean_absolute_error",
    cv=5,
    verbose=1,
    n_jobs=-1,
)
grid_search_pca.fit(X_train, y_train)

mejores_parametros_lasso_pca = grid_search_pca.best_params_
y_pred_pca = grid_search_pca.predict(X_test)

mae_pca = mean_absolute_error(y_test, y_pred_pca)
mape_pca = mean_absolute_percentage_error(y_test, y_pred_pca)
mse_pca = mean_squared_error(y_test, y_pred_pca)
r2_pca = r2_score(y_test, y_pred_pca)

print(f"Mejores hiperparámetros con PCA: {mejores_parametros_lasso_pca}")
print(f"MAE con PCA: {round(mae_pca, 2)}")
print(f"MAPE con PCA: {round(mape_pca, 2)} %")
print(f"R2 con PCA: {round(r2_pca, 2)}")
