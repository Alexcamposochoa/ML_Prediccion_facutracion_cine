import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    PowerTransformer,
    PolynomialFeatures,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from scipy.stats import boxcox
from scipy import stats
import pickle
from tensorflow import keras


with open("final_model.pkl", "rb") as archivo_entrada:
    final_model = pickle.load(archivo_entrada)

df = pd.read_csv("../data/test/test.csv")

results = final_model.predict(df)
print(results)
