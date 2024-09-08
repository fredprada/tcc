import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import pickle
import requests
from io import StringIO, BytesIO
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw
from sklearn.cluster import DBSCAN
import joblib

# Configurar o layout para "wide"
st.set_page_config(layout="wide")

# Função para carregar um objeto em pickle do GitHub
def load_object_from_github(file_url, github_token):
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3.raw"
    }
    response = requests.get(file_url)#, headers=headers)
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
    return pickle.loads(response.content)

# Função para carregar um arquivo CSV do GitHub
def load_csv_from_github(file_url, github_token):
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3.raw"
    }
    response = requests.get(file_url)#, headers=headers)
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
    return pd.read_csv(StringIO(response.text))

# URL dos arquivos no GitHub
base_url = "https://raw.githubusercontent.com/fredprada/tcc/main/"
model_xgb_cooler_url = base_url + "model_xgb_cooler.pkl"
model_xgb_valve_url = base_url + "model_xgb_valve.pkl"
model_xgb_leakage_url = base_url + "model_xgb_leakage.pkl"
model_xgb_accumulator_url = base_url + "model_xgb_accumulator.pkl"
scaler_url = base_url + "scaler.pkl"
test_data_url = base_url + "x_test_final.csv"
df_tratado_pd_not_optimal_30_rand_instances_url = base_url + "df_tratado_pd_not_optimal_30_rand_instances.csv"
df_sintetico_concatenado_url = base_url + "df_sintetico_concatenado.csv"
df_sintetico_concatenado_sem_scaler_url = base_url + "df_sintetico_concatenado_sem_scaler.csv"

# Token de acesso ao GitHub
GITHUB_TOKEN = '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'

# Carregar os modelos e o scaler
model_xgb_cooler = load_object_from_github(model_xgb_cooler_url, GITHUB_TOKEN)
model_xgb_valve = load_object_from_github(model_xgb_valve_url, GITHUB_TOKEN)
model_xgb_leakage = load_object_from_github(model_xgb_leakage_url, GITHUB_TOKEN)
model_xgb_accumulator = load_object_from_github(model_xgb_accumulator_url, GITHUB_TOKEN)
scaler = load_object_from_github(scaler_url, GITHUB_TOKEN)

# Carregar os dados de teste
test_data = load_csv_from_github(test_data_url, GITHUB_TOKEN)
df_tratado = load_csv_from_github(df_tratado_pd_not_optimal_30_rand_instances_url, GITHUB_TOKEN)
df_sintetico_concatenado = load_csv_from_github(df_sintetico_concatenado_url, GITHUB_TOKEN)
df_sintetico_concatenado_sem_scaler = load_csv_from_github(df_sintetico_concatenado_sem_scaler_url, GITHUB_TOKEN)

# Interface do Streamlit
st.title('Predição de Falhas')

# Lista de instâncias
lista_instancias = list(range(1, 41))

# Selecionar instâncias para teste
instancias_para_teste = st.multiselect(
    "Qual instância deseja ver?", 
    options=['Selecionar Tudo'] + lista_instancias, 
    default=[1]
)

# Tratar a opção "Selecionar Tudo"
if 'Selecionar Tudo' in instancias_para_teste:
    instancias_para_teste = list(range(1, 11))

if len(instancias_para_teste) == 0:
    st.write("Selecione pelo menos uma instância para visualizar os dados.")
else:
    max_ciclo = 60
    contador_placeholder = st.empty()

    # Criar espaços reservados para gráficos
    cols = st.columns(4)
    placeholders = [col.empty() for col in cols]

    if st.button("Start"):
        for num_ciclos in range(1, max_ciclo + 1):
            time.sleep(1)  # Simular um delay de 1 segundo para cada ciclo

            # Filtrar os dados com base no número de ciclos
            X_test_pivoted_with_results = df_sintetico_concatenado_sem_scaler[
                (df_sintetico_concatenado_sem_scaler['id'].isin(instancias_para_teste)) & 
                (df_sintetico_concatenado_sem_scaler['ciclo_sequencial'] <= num_ciclos)
            ]

            # Atualizar os gráficos dinamicamente organizando-os 4 por linha
            for idx, sensor in enumerate(lista_sensores):
                df_filtrado_sensor = X_test_pivoted_with_results[['ciclo_sequencial', 'id', sensor]].rename(columns={sensor: 'valor', 'ciclo_sequencial': 'ciclo'})

                # Criar um gráfico Altair com interatividade
                chart = alt.Chart(df_filtrado_sensor).mark_line().encode(
                    x='ciclo',
                    y=alt.Y('valor', title=f'Valor ({unidades_sensores[idx]})'),
                    color=alt.Color('id:N', legend=alt.Legend(title="Instância")),
                    tooltip=['id', 'ciclo', 'valor']
                ).properties(
                    title=f'{nomes_sensores[idx]}'
                ).interactive()  # Permite zoom e pan

                # Determinar a linha e coluna para o gráfico
                with cols[idx % 4]:
                    placeholders[idx % 4].altair_chart(chart, use_container_width=True)
