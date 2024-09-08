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

# Lista de sensores
lista_sensores = ['ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'eps1', 'fs1', 'fs2', 'ts1', 'ts2', 'ts3', 'ts4', 'vs1', 'ce', 'cp', 'se']
nomes_sensores = [
    'Pressão 1', 'Pressão 2', 'Pressão 3', 'Pressão 4', 'Pressão 5', 'Pressão 6',
    'Potência do motor', 'Fluxo 1', 'Fluxo 2', 'Temperatura 1', 'Temperatura 2',
    'Temperatura 3', 'Temperatura 4', 'Vibração 1', 'Eficiência do Resfriador', 
    'Potência do Resfriador', 'Fator de eficiência'
]
unidades_sensores = ['bar', 'bar', 'bar', 'bar', 'bar', 'bar', 'W', 'l/min', 'l/min', '°C', '°C', '°C', '°C', 'mm/s', '%', 'kW', '%']

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

    if st.button("Start"):
        for num_ciclos in range(1, max_ciclo + 1):
            time.sleep(1)  # Simular um delay de 1 segundo para cada ciclo

            # Filtrar os dados com base no número de ciclos
            X_test_pivoted_with_results = df_sintetico_concatenado_sem_scaler[
                (df_sintetico_concatenado_sem_scaler['id'].isin(instancias_para_teste)) & 
                (df_sintetico_concatenado_sem_scaler['ciclo_sequencial'] <= num_ciclos)
            ]

            # Organizar 4 gráficos por linha
            num_sensores = len(lista_sensores)
            for i in range(0, num_sensores, 4):
                cols = st.columns(4)  # Criar uma nova linha com 4 colunas
                for j, sensor in enumerate(lista_sensores[i:i+4]):
                    df_filtrado_sensor = X_test_pivoted_with_results[['ciclo_sequencial', 'id', sensor]].rename(columns={sensor: 'valor', 'ciclo_sequencial': 'ciclo'})

                    # Definir o intervalo fixo para o eixo y
                    y_max = df_sintetico_concatenado_sem_scaler[sensor].max() * 1.1
                    y_min = df_sintetico_concatenado_sem_scaler[sensor].min() * 0.9

                    # Criar um gráfico Altair com interatividade
                    chart = alt.Chart(df_filtrado_sensor).mark_line().encode(
                        x='ciclo',
                        y=alt.Y('valor', title=f'Valor ({unidades_sensores[i+j]})', scale=alt.Scale(domain=[y_min, y_max])),
                        color=alt.Color('id:N', legend=alt.Legend(title="Instância")),
                        tooltip=['id', 'ciclo', 'valor']
                    ).properties(
                        title=f'{nomes_sensores[i+j]}'
                    ).interactive()  # Permite zoom e pan

                    # Exibir o gráfico na coluna correta
                    cols[j].altair_chart(chart, use_container_width=True)
