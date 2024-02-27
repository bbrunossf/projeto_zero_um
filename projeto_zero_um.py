# ==================================================
# Bibliotecas Necessárias
# ==================================================
import sqlite3
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image 
from sklearn import tree
from sklearn import tree as tr
from sklearn.tree import DecisionTreeClassifier

#-------------------------------------Início das Funções-----------------------------------

st.set_page_config(page_title='Projeto Zero Um', page_icon='💾', layout='wide') 

# ==================================================
# Import dataset
# ==================================================
conn=sqlite3.connect("database.db")

consulta_atividade="""

    SELECT* 
    FROM flight_activity fa LEFT JOIN flight_loyalty_history flh ON (fa.loyalty_number=flh.loyalty_number)
    
    """
    
df_atividade=pd.read_sql_query(consulta_atividade, conn)

# ==================================================
# Selecionando Somente as Colunas que Contêm Números
# ==================================================
colunas=["year", "month", "flights_booked", "flights_with_companions", "total_flights", "distance",
         "points_accumulated", "salary", "clv", "loyalty_card"]

df_colunas_numericas=df_atividade.loc[:, colunas]

# ==================================================
# Limpando os dados
# ==================================================
df_dados_completos=df_colunas_numericas.dropna()

# ==================================================
# Layout no Streamliy
# ==================================================
st.title('Propensão de Compra do Cliente')
st.markdown(
    """ 
    ##### O objetivo principal e traçar perfis individuais para cada clientes com base em seus programas de fidelidade de voos:  Star, Nova e Aurora. 
    ##### A partir desses perfis, foi desenvolvido um modelo capaz de recomendar o programa de fidelidade mais adequado para novos clientes, estimando suas probabilidades de adesão a cada opção.
""")
st.subheader('', divider='gray') 

# ==================================================
# Machine Learning
# ==================================================
x_atributos=df_dados_completos.drop(columns="loyalty_card") 

y_rotulos=df_dados_completos.loc[:, "loyalty_card"]

# Definição do Algoritimo
modelo=tr.DecisionTreeClassifier(max_depth=3) 

# Treinamento do Algoritimo
modelo_treinado=modelo.fit(x_atributos, y_rotulos)

tr.plot_tree(modelo_treinado, filled=True);

# ==================================================
# Barra Lateral
# ==================================================
image=Image.open('logo.png')
st.sidebar.image(image, width=250)

st.sidebar.title('Atributos dos Clientes')
st.sidebar.subheader('', divider='gray')

st.sidebar.subheader('Selecione o(s) slider(s)  que deseja analisar:')

def predict():
    
    year                    = st.sidebar.slider("Year", 2017, 2018, step=1)
    salary                  = st.sidebar.slider("Salary", 58486.00, 407228.00, step=0.1)
    clv                     = st.sidebar.slider("Clv", 2119.89, 83325.38, step=0.1)
    month                   = st.sidebar.slider("Month", 1, 12, step=1)
    flights_booked          = st.sidebar.slider("Flights_booked", 0, 21, step=1)
    flights_with_companions = st.sidebar.slider("Flights_with_companions", 0, 11, step=1)
    total_flights           = st.sidebar.slider("Total_flights", 0, 32, step=1)
    distance                = st.sidebar.slider("Distance", 0, 6293, step=1)
    points_accumulated      = st.sidebar.slider("Points_accumulated", 0.00, 676.50, step=0.1)
          
    x_novo=np.array([[year, month, flights_booked, flights_with_companions, 
                          total_flights, distance, points_accumulated, salary, clv]])

    previsao = modelo_treinado.predict_proba(x_novo)

    return {"Aurora": previsao[0][0], "Nova": previsao[0][1], "Star": previsao[0][2]}


# ==================================================
# Visualização dos Resultados
# ==================================================
user = predict()

st.markdown('#### Programa de Fidelidade Recomendado')

melhor_programa = max(user, key=user.get)
cores = {'Aurora': '#0000FF', 'Nova': '#76C5F7', 'Star': '#FF0000'}
texto_formatado = f"##### <span style='color:{cores[melhor_programa]}'>**{melhor_programa}: {user[melhor_programa]*100:.2f}%**</span>"
st.markdown(texto_formatado, unsafe_allow_html=True)

st.subheader('', divider='gray')


df_user = pd.DataFrame.from_dict(user, orient='index', columns=['previsao']).reset_index().rename(columns={'index': 'Programa'})

fig = px.bar(df_user, x='Programa', y='previsao', color='Programa', title='Probabilidades de Adesão ao Programa de Fidelidade', text_auto=True,
             labels={'Programa': 'Programa', 'previsao': 'Probabilidade'})

fig.update_traces(textfont_size=15, textangle=1, textposition="outside", cliponaxis=False, textfont_color='black')

fig.update_layout(yaxis_tickformat='.2%', yaxis_title='Probabilidade (%)')

st.plotly_chart(fig)


st.sidebar.subheader('', divider='gray')                
st.sidebar.subheader('Powered by: Jadson N Santos')
st.sidebar.subheader('Discord: jadson')
st.sidebar.subheader('Linkedin: https://www.linkedin.com/in/jadson-nascimento-santos/')
st.sidebar.subheader('GitHub: https://github.com/JadsonDS') 
st.sidebar.subheader('Portfolio de Projetos: https://jadsonds.github.io/portfolio_projetos/')
