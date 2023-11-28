import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as c

st.set_page_config(page_title='Alex Campos ML')

st.title('_¿Cuanto facturaré con esta película?_')
george_lucas = Image.open('images/george_lucas.webp')
st.image(george_lucas,use_column_width=True)


processed = pd.read_csv('data/processed/processed.csv')

seleccion = st.sidebar.selectbox("Selecciona menu", ['Home','Datos','Directores',])

if seleccion == "Home":
    st.title("Cargadores Madrid")
    img = Image.open("data/puntos-recarga-madrid.jpg")
    st.image(img)
    with st.expander("Introducción"):
        st.write("Es una solución factible para empezar a trasicionar a un tipo de energías más limpias en cuanto a emisiones en la ciudad")

    with st.expander("Tabla"):
        df = pd.read_csv("data/red_recarga_acceso_publico.csv", sep=";")
        st.write(df.head())

elif seleccion == "Datos":

    df = pd.read_csv("data/red_recarga_acceso_publico.csv", sep=";")

    filtro = st.sidebar.selectbox("Selecciona un distrito", df['DISTRITO'].unique())
    df_filtered = df[df['DISTRITO']==filtro]
    st.write(df_filtered)

    file = open("data/heatmap.html", "r")
    c.html(file.read(), height=400)

    df_filtered.rename(columns={"latidtud":"lat", "longitud":"lon"}, inplace=True)
    # st.write(df)

    st.map(df_filtered)

    filtro_2 = st.sidebar.radio("Elige el nº de cargadores", [1,2,3,4])

    st.sidebar.button("Click aquí")