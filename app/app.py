import streamlit as st
from PIL import Image
import streamlit.components.v1 as c
import numpy as np
import pandas as pd
import pickle


# configuración de la página

st.set_page_config(
    page_title="Alex Campos ML", initial_sidebar_state="collapsed", layout="wide"
)

# with open(os.path.join(dir_path, "..", "models", "modelo_lineal", "trained_pol_3.pkl"),"rb") as li:
#     lin_reg = pickle.load(li)

with open("recursos/final_model.pkl", "rb") as file:
    final_model = pickle.load(file)


# creación barra lateral

seleccion = st.sidebar.selectbox(
    "Selecciona menu",
    ["Bienvenida", "Data Science", "Negocio", "Haz tu predicción"],
)

# flujo del menú lateral
# creamos página de "Bienvenida"

if seleccion == "Bienvenida":
    # left_co, cent_co, last_co = st.columns(3)
    # with cent_co:
    st.title("""Bienvenidos a Predicting Success""")
    st.divider()
    logo = Image.open("recursos/logo.jpg")
    # left_co, cent_co, last_co = st.columns(3)
    # with cent_co:
    st.image(logo, width=1400)
    st.divider()
    st.subheader("_Un Viaje Analítico a través de las Películas con Machine Learning_")
    intro_marketing = """
    En esta ocasión descubriremos el poder del Machine Learning en el mundo del cine.

    Sumérgete en un fascinante viaje a través del séptimo arte con nuestro proyecto de machine learning dedicado al análisis cinematográfico. En este emocionante recorrido, exploraremos el universo de las películas, desentrañando secretos y patrones ocultos que influyen en su éxito financiero.

    Imagina tener la capacidad de predecir el rendimiento de una película antes de su lanzamiento, utilizando avanzadas técnicas de aprendizaje automático. En esta ocasión en Predicting Sucess, nos sumergimos en la rica fuente de datos que ofrece el mundo del cine para anticipar el desempeño financiero, centrándonos en la variable clave: la facturación (Gross).

    Acompáñanos mientras analizamos atributos como director, estrella principal, presupuesto, puntuación de la crítica, fecha de lanzamiento, género y más. Estamos emocionados de compartir contigo los descubrimientos y predicciones que hemos destilado de vastos conjuntos de datos cinematográficos.

    """
    st.write(intro_marketing)
    st.divider()

# creamos página de "Data Science"

if seleccion == "Data Science":
    cientificos = Image.open("recursos/cientificos_lab.jpg")
    st.title("Memoria técnica")
    st.divider()
    st.image(cientificos, width=1400)
    st.divider()
    with st.expander("Introducción y plan"):
        st.write(
            """
Se pide la elaboración de un proyecto de machine learning de temática libre, en el cual demostraremos gran parte de las competencias y herramientas adquiridas tanto en analisis como en aprendizaje automatico.

La temática elegida es sobre el mundo del cine y las películas. Existe una abundante cantidad de información y datos disponibles en la red, susceptibles de ser analizados para la creación de un modelo de predicción.  

Nuestra variable a predecir en este caso será la facturación, denominada en adelante como 'Gross', y las variables predictoras iniciales incluyen atributos comunes a todas las películas, tales como director, estrella principal, presupuesto, puntuación de la crítica, fecha de lanzamiento, género, entre otros.              """
        )
        plan = Image.open("recursos/plan.png")
        st.image(plan)  # , width=1400)
    with st.expander("Dataset Original"):
        df = pd.read_csv("recursos/movies.csv")
        st.write(df.head(7))
        st.write("7668 rows y 15 columns")
        st.divider()
        info_output = df.info()
        st.header("Información del DataFrame:")
        # st.code("""df.info()""")
        st.write("df.describe()")
        st.write(df.describe())
        st.divider()
        st.header("Conclusiones iniciales")
        st.write(
            """
                - Gran parte de nuestras variables son texto y el feature engineering deberá ser exhaustivo 
                - Tenemos películas desde los año 80 a 2020
                - La std de algunas variables es considerablemente grande
                - Estudiar outliers en budget y gross"""
        )
        st.divider()

    with st.expander("EDA inicial"):
        st.write("Distribución de la variable target")
        distribucion_target = Image.open("recursos/distribucion_target_incial.png")
        # left_co, cent_co, last_co = st.columns(3)
        # with cent_co:
        st.image(distribucion_target)
        st.divider()

        st.write("Scatters de distribucion")
        # left_co, cent_co, last_co = st.columns(3)
        scatters = Image.open("recursos/scatters.png")
        # with cent_co:
        st.image(scatters)
        st.divider()
        st.write("Heatmap")
        heatmap = Image.open("recursos/heatmap_inicial.png")
        # left_co, cent_co, last_co = st.columns(3)
        # with cent_co:
        st.image(heatmap)
        st.divider()

    with st.expander("Feature engineering y limpieza de datos"):
        st.title("Primeros pasos")
        st.write(
            """Tras el heatmap, el breve EDA inicial y el conocimiento de negocio, establecemos una jerarquia entre las variables para elegir cuales mapear y decidir una importancia
                 mas controlada y cuales trabajar con una etiqueta automática de label encoder"""
        )
        st.write("Mapeos")
        st.code(
            """mapeo_rating =
            {'Unrated':0,
            'Not Rated':1,
            'Approved':2,
            'X':3,
            'NC-17':4,
            'TV-MA':5,
            'R':6,
            'TV-14':7,
            'PG-13':8,
            'G':9,
            'PG':10}

            df['rating_mapped'] = df['rating'].map(mapeo_rating)"""
        )
        st.divider()
        st.code(
            """mapeo_genre =
            {'Action':19,
            'Adventure':18,
            'Comedy':17,
            'Horror':16,
            'Drama':15,
            'Thriller':14,
            'Romance':13,
            'Musical':12,
            ...
               
            df['genre_mapped'] = df['genre'].map(mapeo_genre)"""
        )
        st.divider()
        st.code(
            """mapeo_directores=
            {'Stanley Kubrick': 9,
            'Randal Kleiser': 5,
            'Irvin Kershner': 6,
            'Jim Abrahams': 4,
            'Harold Ramis': 7,
            'Sean S. Cunningham': 3,
            'John Landis': 6,
            'Martin Scorsese': 8,
            'Richard Lester': 5,

            ...2949 directores totales...}
                
            df['director_mapped'] = df['director'].map(mapeo_directores)"""
        )
        st.divider()
        st.code(
            """mapeo_star =
            {'Jack Nicholson': 10,
            'Brooke Shields': 6,
            'Mark Hamill': 8,
            'Robert De Niro': 10,
            'Gene Hackman': 9,
            'David Carradine': 7,
            'Clint Eastwood': 10,
            'N!xau': 6,
            'Robin Williams': 10}
            ...2814 star totales...
                
            df['star_mapped'] = df['star'].map(mapeo_star)
            """
        )
        st.divider()
        st.write("Regex")
        st.code("""df['country_date'] = df['released'].str.extract(r'\((.*?)\)')""")
        st.divider()
        st.write("Label Encoder")
        st.code(
            """
            le = LabelEncoder()
            writer = df['writer']
            writer_codif = le.fit_transform(writer)
            df['writer_codif'] = le.fit_transform(df['writer'])

            le = LabelEncoder()
            country = df['country']
            country_codif = le.fit_transform(country)
            df['country_codif'] = le.fit_transform(df['country'])

            le = LabelEncoder()
            company = df['company']
            company_codif = le.fit_transform(company)
            df['company_codif'] = le.fit_transform(df['company'])

            le = LabelEncoder()
            name = df['name']
            name_ID = le.fit_transform(name)
            df['name_ID'] = le.fit_transform(df['name'])
            """
        )
        st.divider()
        st.write("Creación de variables propias")
        st.code(
            """
            df['director_mapped'] = df['director_mapped'] * 10
            df['star_mapped'] = df['star_mapped'] * 10
            df['votos_por_director'] = df.groupby('director_mapped')['votes'].transform('mean').round(2)
            df['facutracion_por_director'] = df.groupby('director_mapped')['gross'].transform('mean').round(2)
            df['votos_por_año'] = df.groupby('year')['votes'].transform('mean').round(2)
            df['facturacion_por_genre'] = df.groupby('genre_mapped')['gross'].transform('mean').round(2)
            
                """
        )
        st.divider()
        st.write("Dataset numérico")
        dfnumeros = pd.read_csv("recursos/df_numerico_head.csv")
        st.write(dfnumeros)
        st.divider()

    with st.expander("Transformación de datos"):
        st.divider()
        st.subheader("Year")
        st.write(
            "Durante alguna parte del proceso iterativo incremental apreciamos que en algunas variables los datos están muy desviados debido a la gran variedad de la variable YEAR con lo que decidimos trabajar solo con los datos del año 2000 en adelante"
        )
        st.code("""df= df[df['year'] >= 2000]""")
        st.divider()
        st.subheader("Genre")
        st.write(
            "La variable genero estaba trabajada con un mapeo erroneo y apenas estaba aportando correlación decidimos reducir el numero de valores agrupando en otros generos las menos importantes"
        )
        st.code(
            """
            df['genre_mapped'].value_counts()
                
            genre_mapped
            17.0    1002
            19.0    1000
            15.0     791
            6.0      278
            9.0      246
            18.0     210
            16.0     174
            10.0      13
            8.0        8
            14.0       5
            13.0       4
            11.0       4
            1.0        3
            12.0       2
            4.0        1
            Name: count, dtype: int64

            df['genre_mapped'] = df['genre_mapped'].apply(lambda x: x if 14 <= x <= 19 else 13)

                """
        )
        st.divider()
        st.subheader("Tratamiento de missing values")
        st.write(
            "Consideramos eliminar totalmente los registros con nans debido a la abundante cantidad de datos y la no necesidad de falsear los datos de variables tan importantes como el budget "
        )
        st.code("""df.dropna(inplace=True)""")
        st.divider()
        st.subheader("Correción y transformación de distribuciones")
        st.write(
            "Llegamos este punto en el que la transformación logaritmica Box-Cox nos ayuda a cambiar la forma de nuestros datos y conseguir una distribución normal, nos planteamos si es necesaria la implementación de esta transformación a alguna variable más e incluso todo el dataset."
        )
        st.divider()
        distribucion_gross = Image.open("recursos/distribucion_gross.png")
        st.image(distribucion_gross)
        st.divider()
        distribucion_general = Image.open("recursos/distribuciones_general.png")
        st.image(distribucion_general)
        st.divider()
        distribucion_votes = Image.open("recursos/distribucion_votes.png")
        st.image(distribucion_votes)
        st.divider()
        distribucion_budget = Image.open("recursos/distribucion_budget.png")
        st.image(distribucion_budget)

    with st.expander("Correlación final"):
        heatmap_f = Image.open("recursos/heatmap_final.png")
        st.image(heatmap_f)

    st.title("Modelado")

    with st.expander("Train test split"):
        st.code(
            """X = df.drop(['gross'], axis=1)
y = df['gross']

X_train, X_test, y_train, y_test =
                        train_test_split(X, y,test_size=0.2, random_state=42)"""
        )
        st.code(
            """print(X_train.shape)
print(X_test.shape) 
print(y_train.shape)
print(y_test.shape)

(2187, 12)
(547, 12)
(2187,)
(547,)
"""
        )

    with st.expander("Regresión Lineal"):
        st.code(
            """pipeline_lnr = Pipeline([
    ('scaler', StandardScaler()),  
    ('regression', LinearRegression())
    ])

parametros_grid = {'regression__fit_intercept': [True, False]}

grid_search = GridSearchCV(pipeline_lnr, parametros_grid,
scoring='neg_mean_absolute_error', cv=5, verbose=1)#, n_jobs=-1)

grid_search.fit(X_train, y_train)

mejores_parametros_lnr = grid_search.best_params_

y_pred = grid_search.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mejores hiperparámetros: {mejores_parametros_lnr}")
print(f"MAE: {round(mae, 2)}")
print(f"MAPE: {round(mape, 2)} %")
print(f"R2: {round(r2, 2)}")
                
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Mejores hiperparámetros: {'regression__fit_intercept': True}
MAE: 73591994.34
MAPE: 19.51 %
R2: 0.66   
                
                """
        )

    with st.expander("Regresión Polinomial"):
        st.code(
            """pipeline_polyr = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures()),
    ('regression', LinearRegression())
])

parametros_grid = {
    'poly__degree': [2, 3, 4], 
    'regression__fit_intercept': [True, False]
}

grid_search = GridSearchCV(pipeline_polyr,
    parametros_grid, scoring='neg_mean_absolute_error',
    cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
mejores_parametros_polyr = grid_search.best_params_
y_pred = grid_search.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mejores hiperparámetros: {mejores_parametros_polyr}")
print(f"MAE: {round(mae, 2)}")
print(f"MAPE: {round(mape, 2)} %")
print(f"R2: {round(r2, 2)}")

Fitting 5 folds for each of 6 candidates, totalling 30 fits
Mejores hiperparámetros: {'poly__degree': 4, 'regression__fit_intercept': False}
MAE: 3.1601144926771804e+16
MAPE: 1630418595.44 %
R2: -1.2292022292239306e+20

"""
        )

    with st.expander("Regresión polinomial con regularización Ridge"):
        st.code(
            """pipeline_ridge = Pipeline([
('scaler', StandardScaler()),
('poly', PolynomialFeatures()),
('ridge', Ridge()) 
])

parametros_grid = {
    'poly__degree': [2, 3, 4], 
    'ridge__alpha': [0.1, 1.0, 10.0], 
}

grid_search = GridSearchCV(pipeline_ridge,
            parametros_grid, scoring='neg_mean_absolute_error',
            cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
mejores_parametros_ridge = grid_search.best_params_
y_pred = grid_search.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mejores hiperparámetros: {mejores_parametros_ridge}")
print(f"MAE: {round(mae, 2)}")
print(f"MAPE: {round(mape, 2)} %")
print(f"R2: {round(r2, 2)}")

Fitting 5 folds for each of 9 candidates, totalling 45 fits
Mejores hiperparámetros: {'poly__degree': 2, 'ridge__alpha': 10.0}
MAE: 64347904.35
MAPE: 8.32 %
R2: 0.67

"""
        )

    with st.expander("Regresión polinomial con regularización Lasso"):
        st.code(
            """pipeline_lasso = Pipeline([
    ('scaler', StandardScaler()),  
    ('poly', PolynomialFeatures()),  
    ('lasso', Lasso())  
])

parametros_grid = {
    'poly__degree': [2, 3, 4, 5],  
    'lasso__alpha': [0.1, 1.0, 10.0], 
}

grid_search = GridSearchCV(pipeline_lasso, parametros_grid,
        scoring='neg_mean_absolute_error', cv=5, verbose=1,
        n_jobs=-1)
grid_search.fit(X_train, y_train)
mejores_parametros_lasso = grid_search.best_params_
y_pred = grid_search.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mejores hiperparámetros: {mejores_parametros_lasso}")
print(f"MAE: {round(mae, 2)}")
print(f"MAPE: {round(mape, 2)} %")
print(f"R2: {round(r2, 2)}")

Fitting 5 folds for each of 12 candidates, totalling 60 fits
Mejores hiperparámetros: {'lasso__alpha': 10.0, 'poly__degree': 2}
MAE: 66968519.14
MAPE: 7.89 %
R2: 0.64

"""
        )

    with st.expander("Regresión polinomial con Elastic Net"):
        st.code(
            """pipeline_en = Pipeline([
    ('scaler', StandardScaler()), 
    ('poly', PolynomialFeatures()), 
    ('elastic_net', ElasticNet())
])

parametros_grid = {
    'poly__degree': [2, 3, 4], 
    'elastic_net__alpha': [0.1, 1.0, 10.0], 
    'elastic_net__l1_ratio': [0.1, 0.5, 0.9]
}

grid_search = GridSearchCV(pipeline_en, parametros_grid,
        scoring='neg_mean_absolute_error', cv=5, verbose=1,
        n_jobs=-1)
grid_search.fit(X_train, y_train)
mejores_parametros_en = grid_search.best_params_
y_pred = grid_search.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mejores hiperparámetros: {mejores_parametros_en}")
print(f"MAE: {round(mae, 2)}")
print(f"MAPE: {round(mape, 2)} %")
print(f"R2: {round(r2, 2)}")

Fitting 5 folds for each of 27 candidates, totalling 135 fits
Mejores hiperparámetros: {'elastic_net__alpha': 0.1, 'elastic_net__l1_ratio': 0.9, 'poly__degree': 2}
MAE: 63460778.31
MAPE: 8.8 %
R2: 0.68

"""
        )

    with st.expander("Random Forest Regressor"):
        st.code(
            """pipeline_rf = Pipeline([
    ('scaler', StandardScaler()), 
    ('rf', RandomForestRegressor(random_state=42))
])

param_grid = {
    'rf__n_estimators': [5, 10, 20],  
    'rf__max_depth': [None, 10, 20],  
}

grid_search = GridSearchCV(pipeline_rf, param_grid, cv=5,
        scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_rf = best_model.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regressor:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"MAE: {round(mae_rf, 2)}")
print(f"MAPE: {round(mape, 2)} %")
print(f"R2: {round(r2_rf, 2)}")

Random Forest Regressor:
Best Parameters: {'rf__max_depth': 10, 'rf__n_estimators': 20}
MAE: 59139430.25
MAPE: 8.8 %
R2: 0.7

"""
        )

    with st.expander("Gradient Boosting"):
        st.code(
            """pipeline_gb = Pipeline([
('scaler', StandardScaler()), 
('gb', GradientBoostingRegressor(random_state=42))
])


param_grid_gb = {
'gb__n_estimators': [50, 100, 200], 
'gb__learning_rate': [0.01, 0.1, 0.2], 
'gb__max_depth': [3, 5, 7],
}

grid_search_gb = GridSearchCV(pipeline_gb, param_grid_gb,
    cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

best_model_gb = grid_search_gb.best_estimator_

y_pred_gb = best_model_gb.predict(X_test)

mae_gb = mean_absolute_error(y_test, y_pred_gb)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2_gb = r2_score(y_test, y_pred_gb)

print("Gradient Boosting Regressor:")
print(f"Best Parameters: {grid_search_gb.best_params_}")
print(f"MAE: {round(mae_gb, 2)}")
print(f"MAPE: {round(mape, 2)} %")
print(f"R2: {round(r2_gb, 2)}")

Gradient Boosting Regressor:
Best Parameters: {'gb__learning_rate': 0.1, 'gb__max_depth': 5, 'gb__n_estimators': 100}
MAE: 58732665.82
MAPE: 8.8 %
R2: 0.64
        



        """
        )

    with st.expander("Métricas, evaluación y decisión final"):
        st.write(
            """La métrica en la que queremos basarnos para la eleccion del modelo es el Mape debido a que estamos hablando de predecir cientos de miles de dolares y creemos que resulta mas entendible manejar porcentajes de error."""
        )
        metricas = Image.open("recursos/metricas.png")
        st.image(metricas)

        st.divider()
        st.write(
            """
Finalmente el modelo elegido es la Regresión Polinomial con regularización Lasso y PCA.

En conjunto, la regresión polinómica captura complejidades en la relación entre las variables, Lasso trabaja con la regularización y nos ayuda a evitar el sobreajuste al eliminar características menos importantes y el PCA reduce puede reducir la dimensionalidad del modelo y conseguir con esta combinación un modelo robusto y eficiente."""
        )

        st.code(
            """pipeline_lasso_pca = Pipeline([
    ('scaler', StandardScaler()),  
    ('poly', PolynomialFeatures()),  
    ('pca', PCA()), 
    ('lasso', Lasso())  
])

parametros_grid_pca = {
    'poly__degree': [2],  
    'pca__n_components': [None],  
    'lasso__alpha': [10.0], 
}

grid_search_pca = GridSearchCV(pipeline_lasso_pca, parametros_grid_pca, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid_search_pca.fit(X, y)

mejores_parametros_lasso_pca = grid_search_pca.best_params_

modelo_final = grid_search_pca.best_estimator_
modelo_final.fit(X, y)
                

Mejores hiperparámetros con PCA: {'lasso__alpha': 10.0, 'pca__n_components': None, 'poly__degree': 2}
MAE con PCA: 67489365.6
MAPE con PCA: 7.85 %
R2 con PCA: 0.64"""
        )

        st.divider()

        st.write(
            "Aquí podemos visualizar el recorrido de la curva de aprendizaje, es decir podemos ver si añadiendo mas datos de entrenamiento el error en validación mejoraría. La conclusión de esta visualización es que hemose seleccionado un punto en el que no va a mejorar mas y es probable que con mas datos de entrenamiento caiga en overfiting."
        )

        curva = Image.open("recursos/curva_aprendizaje.png")
        st.image(curva)

        st.divider()

        importancia = Image.open("recursos/feature_importance.png")
        st.image(importancia)

        st.divider()

        st.write(
            "Con este scatter podemos ver de manera clara como se dispersan las predicciónes frente los valores reales, cuanto más claros estarán mas cercanos a la linea negra que representa los valores reales y cuanto más coloreados serán predicciones positivas o negativas mas dispersos o alejados de los datos reales."
        )

        dispersion = Image.open("recursos/dispersion_prediccion.png")
        st.image(dispersion)

        st.divider()

    # elif seleccion == "Cliente":
    # df = pd.read_csv("app/recursos/red_recarga_acceso_publico.csv", sep=";")

    # filtro = st.sidebar.selectbox("Selecciona un distrito", df["DISTRITO"].unique())
    # df_filtered = df[df["DISTRITO"] == filtro]
    # st.write(df_filtered)

    # file = open("data/heatmap.html", "r")
    # c.html(file.read(), height=400)

    # df_filtered.rename(columns={"latidtud": "lat", "longitud": "lon"}, inplace=True)
    # # st.write(df)

    # st.map(df_filtered)

    # filtro_2 = st.sidebar.radio("Elige el nº de cargadores", [1, 2, 3, 4])

    # st.sidebar.button("Click aquí")

    with st.expander("Lecciones aprendidas"):
        st.write(
            """Este trabajo ha sido la concatenación de muchas herramientas obtenidas en relativamente poco tiempo. Frustrante en muchas ocasiones ya que el propio aprendizaje lleva intrínseco la visualización de expcionales trabajos y es muy dificil evitar la comparación con uno mismo , olvidando al 100% que somos estudiantes y estamos aprendiendo.

Dicho esto, mis conclusiones sobre las lecciones aprendidas son:

- La dinámica iterativa incremental es una manera maravillosa de no atascarse y afrontar los problemas a su debido momento, cuando ya existe o has creado un contexto para poder abordar algo con mayor profundidad de negocio.

- He concluido (quizá erroneamente,y si es así por favor iluminenme) que en proyectos donde el dataset no esté limpio el porcentaje de tiempo dedicado a la compresion del dataset inicial, al feature engineering, limpieza de datos y creación de nuevas variables propias es practicamente lo más importante y lo que mejores resultados te brindará. Pero en esta ocasión la dimension del proyecto y la prisa por hacer lo "guay" (modelos de ML) ocultaron la realidad y quizá el margen de mejora estaría ahí.

- A pesar de que me ha gustado mucho poder dedicar estas 3 semanas a un proyecto, los resultados no son lo que a mi me gustaría, siendo crítico y haciendo una prueba con datos reales, y teniendo la predicción el modelo no se acerca a lo que en realidad ha sucedido y eso es desalentador, o bien por mi falta de capacidad a la hora de realizar el proyecto o bien por que esto del machine learning como dice un conocido es pseudociencia. """
        )
        st.divider()
        st.write(
            "Gracias a profesores y a todos mis compañeros, ya que sin su ayuda estoy seguro que esto no hubiese salido, o por lo menos igual."
        )
        st.divider()
        st.write(
            """Proyecto elaborado por Alejandro Campos Ochoa - DS2309 - 

29/11/2023"""
        )
elif seleccion == "Haz tu predicción":
    st.title("¡¡Haz tu predicción!!")
    st.divider()

    prediction = []

    # Añadimos el año

    year = st.slider("Selecciona el año de lanzamiento de tu película", 2000, 2025)
    st.write("Lanzamiento el año:", year)
    prediction.append(year)

    st.divider()

    # Añadimos el score

    score = st.slider("Selecciona la puntuación de la critica", 0.0, 10.0)
    st.write("Score:", score)
    prediction.append(score)

    st.divider()

    # Añadimos los votos

    votes = st.number_input("Cuantos votos ha obtenido tu película de la critica?")
    st.write("Ha obtenido", votes)
    prediction.append(votes)

    st.divider()

    # Añadimos el runtime

    runtime = st.number_input("Cuantos minutos dura tu película?")
    st.write("Duración", runtime)
    prediction.append(runtime)

    st.divider()

    # Añadimos el genero

    genero = st.selectbox(
        "Cual es el genero de tu película?",
        (
            "Action",
            "Adventure",
            "Comedy",
            "Horror",
            "Drama",
            "Thriller",
            "Otros generos",
        ),
    )

    st.write("Tu pelícua es de:", genero)

    if genero == "Action":
        prediction.append(19)
    elif genero == "Adventure":
        prediction.append(18)
    elif genero == "Comedy":
        prediction.append(17)
    elif genero == "Horror":
        prediction.append(16)
    elif genero == "Drama":
        prediction.append(15)
    elif genero == "Thriller":
        prediction.append(14)
    else:
        prediction.append(13)

    st.divider()

    # Añadimos el director

    director = st.selectbox(
        "Que director dirige tu película?",
        (
            "Stanley Kubrick",
            "Steven Spielberg",
            "Denis Villeneuve",
            "Martin Scorsese",
            "Woody Allen",
            "Clint Eastwood",
            "Francis Ford Coppola",
            "Ridley Scott",
            "James Cameron",
            "Hayao Miyazaki",
            "J.J. Abrams",
            "Anthony Russo",
            "Ben Affleck",
            "Richard LaGravenese",
            "Richard Curtis",
            "Francis Lawrence",
            "Kevin Spacey",
            "Olivia Wilde",
            "Christophe Barratier",
            "David Slade",
            "Tony Gilroy",
            "Ryan Fleck",
            "Chris Wedge",
        ),
    )

    if director == "Stanley Kubrick":
        prediction.append(90)
    elif director == "Steven Spielberg":
        prediction.append(80)
    elif director == "Denis Villeneuve":
        prediction.append(80)
    elif director == "Martin Scorsese":
        prediction.append(80)
    elif director == "Woody Allen":
        prediction.append(90)
    elif director == "Clint Eastwood":
        prediction.append(80)
    elif director == "Francis Ford Coppola":
        prediction.append(80)
    elif director == "Ridley Scott":
        prediction.append(80)
    elif director == "James Cameron":
        prediction.append(80)
    elif director == "Hayao Miyazaki":
        prediction.append(90)
    elif director == "J.J. Abrams":
        prediction.append(70)
    elif director == "Anthony Russo":
        prediction.append(70)
    elif director == "Ben Affleck":
        prediction.append(70)
    elif director == "Richard LaGravenese":
        prediction.append(70)
    elif director == "Richard Curtis":
        prediction.append(70)
    elif director == "Francis Lawrence":
        prediction.append(80)
    elif director == "Kevin Spacey":
        prediction.append(80)
    elif director == "Olivia Wilde":
        prediction.append(70)
    elif director == "Christophe Barratier":
        prediction.append(80)
    elif director == "David Slade":
        prediction.append(70)
    elif director == "Tony Gilroy":
        prediction.append(70)
    elif director == "Ryan Fleck":
        prediction.append(70)
    elif director == "Chris Wedge":
        prediction.append(70)

    st.divider()

    # Añadimos la star

    star = st.selectbox(
        "Cual será la estrella principal de tu película?",
        (
            "Jack Nicholson",
            "Robert De Niro",
            "Clint Eastwood",
            "Robin Williams",
            "Al Pacino",
            "Tom Hanks",
            "Leonardo DiCaprio",
            "Ewan McGregor",
            "Scarlett Johansson",
            "Matt Damon",
            "Joaquin Phoenix",
            "Adrien Brody",
            "Owen Wilson",
            "Vanessa Paradis",
            "Monica Bellucci",
            "Natalie Portman",
            "Angelina Jolie",
            "Jane Fonda",
            "Kirk Douglas",
            "John Travolta",
            "Robert Redford",
            "Sylvester Stallone",
            "Tom Cruise",
            "Eddie Murphy",
            "Mark Hamill",
            "Donald Sutherland",
            "Barbra Streisand",
        ),
    )

    if star == "Jack Nicholson":
        prediction.append(100)
    elif star == "Robert De Niro":
        prediction.append(80)
    elif star == "Clint Eastwood":
        prediction.append(100)
    elif star == "Robin Williams":
        prediction.append(100)
    elif star == "Al Pacino":
        prediction.append(100)
    elif star == "Tom Hanks":
        prediction.append(80)
    elif star == "Leonardo DiCaprio":
        prediction.append(90)
    elif star == "Ewan McGregor":
        prediction.append(100)
    elif star == "Scarlett Johansson":
        prediction.append(90)
    elif star == "Matt Damon":
        prediction.append(90)
    elif star == "Joaquin Phoenix":
        prediction.append(90)
    elif star == "Adrien Brody":
        prediction.append(90)
    elif star == "Owen Wilson":
        prediction.append(90)
    elif star == "Vanessa Paradis":
        prediction.append(90)
    elif star == "Monica Bellucci":
        prediction.append(80)
    elif star == "Natalie Portman":
        prediction.append(100)
    elif star == "Angelina Jolie":
        prediction.append(100)
    elif star == "Jane Fonda":
        prediction.append(100)
    elif star == "Kirk Douglas":
        prediction.append(100)
    elif star == "John Travolta":
        prediction.append(100)
    elif star == "Robert Redford":
        prediction.append(100)
    elif star == "Sylvester Stallone":
        prediction.append(100)
    elif star == "Tom Cruise":
        prediction.append(100)
    elif star == "Eddie Murphy":
        prediction.append(100)
    elif star == "Mark Hamill":
        prediction.append(100)
    elif star == "Donald Sutherland":
        prediction.append(100)
    elif star == "Barbra Streisand":
        prediction.append(100)

    # añadimos votos por director

    if director == "Stanley Kubrick":
        prediction.append(428702.7)
    elif director == "Steven Spielberg":
        prediction.append(236700.88)
    elif director == "Denis Villeneuve":
        prediction.append(236700.88)
    elif director == "Martin Scorsese":
        prediction.append(236700.88)
    elif director == "Woody Allen":
        prediction.append(428702.7)
    elif director == "Clint Eastwood":
        prediction.append(236700.88)
    elif director == "Francis Ford Coppola":
        prediction.append(236700.88)
    elif director == "Ridley Scott":
        prediction.append(236700.88)
    elif director == "James Cameron":
        prediction.append(236700.88)
    elif director == "Hayao Miyazaki":
        prediction.append(428702.7)
    elif director == "J.J. Abrams":
        prediction.append(149101.61)
    elif director == "Anthony Russo":
        prediction.append(149101.61)
    elif director == "Ben Affleck":
        prediction.append(149101.61)
    elif director == "Richard LaGravenese":
        prediction.append(149101.61)
    elif director == "Richard Curtis":
        prediction.append(149101.61)
    elif director == "Francis Lawrence":
        prediction.append(149101.61)
    elif director == "Kevin Spacey":
        prediction.append(236700.88)
    elif director == "Olivia Wilde":
        prediction.append(236700.88)
    elif director == "Christophe Barratier":
        prediction.append(149101.61)
    elif director == "David Slade":
        prediction.append(236700.88)
    elif director == "Tony Gilroy":
        prediction.append(149101.61)
    elif director == "Ryan Fleck":
        prediction.append(149101.61)
    elif director == "Chris Wedge":
        prediction.append(149101.61)

    st.divider()

    # Añadimos el rating

    rating = st.selectbox(
        "Para que público es tu película?",
        (
            "TV-MA, mayores de 18 años",
            "TV-14, mayores de 14 años",
            "PG-13, mayores de 13 años",
            "Unrated, pendiente de clasificar",
            "G, para todos los publicos",
        ),
    )

    if rating == "TV-MA, mayores de 18 años":
        prediction.append(5)
    elif rating == "TV-14, mayores de 14 años":
        prediction.append(7)
    elif rating == "PG-13, mayores de 13 años":
        prediction.append(8)
    elif rating == "Unrated, pendiente de clasificar":
        prediction.append(0)
    elif rating == "G, para todos los publicos":
        prediction.append(9)

    # añadimos facturación por director

    if director == "Stanley Kubrick":
        prediction.append(2.85545745e08)
    elif director == "Steven Spielberg":
        prediction.append(1.91060304e08)
    elif director == "Denis Villeneuve":
        prediction.append(1.91060304e08)
    elif director == "Martin Scorsese":
        prediction.append(1.91060304e08)
    elif director == "Woody Allen":
        prediction.append(2.85545745e08)
    elif director == "Clint Eastwood":
        prediction.append(1.91060304e08)
    elif director == "Francis Ford Coppola":
        prediction.append(1.91060304e08)
    elif director == "Ridley Scott":
        prediction.append(1.91060304e08)
    elif director == "James Cameron":
        prediction.append(1.91060304e08)
    elif director == "Hayao Miyazaki":
        prediction.append(2.85545745e08)
    elif director == "J.J. Abrams":
        prediction.append(1.18734218e08)
    elif director == "Anthony Russo":
        prediction.append(1.18734218e08)
    elif director == "Ben Affleck":
        prediction.append(1.18734218e08)
    elif director == "Richard LaGravenese":
        prediction.append(1.18734218e08)
    elif director == "Richard Curtis":
        prediction.append(1.18734218e08)
    elif director == "Francis Lawrence":
        prediction.append(1.91060304e08)
    elif director == "Kevin Spacey":
        prediction.append(1.91060304e08)
    elif director == "Olivia Wilde":
        prediction.append(1.18734218e08)
    elif director == "Christophe Barratier":
        prediction.append(1.91060304e08)
    elif director == "David Slade":
        prediction.append(1.18734218e08)
    elif director == "Tony Gilroy":
        prediction.append(1.18734218e08)
    elif director == "Ryan Fleck":
        prediction.append(1.18734218e08)
    elif director == "Chris Wedge":
        prediction.append(1.18734218e08)

    st.divider()

    # Añadimos el presupuesto

    budget = st.number_input("Que presupuesto has tenido para tu película?")
    st.write("Tu presupuesto ha sido de", budget)
    prediction.append(budget)

    st.divider()

    # Añadimos facturación por genero

    if genero == "Action":
        prediction.append(145508600)
    if genero == "Adventure":
        prediction.append(109325200)
    if genero == "Comedy":
        prediction.append(44331870)
    if genero == "Horror":
        prediction.append(47372410)
    if genero == "Drama":
        prediction.append(38930960)
    if genero == "Thriller":
        prediction.append(26935260)
    if genero == "Otros generos":
        prediction.append(23549370)

    facturacion_array = np.array(prediction).reshape(1, -1)
    facturacion = final_model.predict(facturacion_array)

    facturacion_formateada = "{:,.2f}".format(abs(facturacion[0]))

    st.success(
        f"La facturación que obtendrás será de: {facturacion_formateada} millones de $"
    )

    st.divider()
