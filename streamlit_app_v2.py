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

    # Inicialize `num_ciclos` antes do uso
    num_ciclos = 1  # Começa com o ciclo inicial, pode ser ajustado conforme necessário
    
    # Criar quatro colunas para os gráficos
    col1, col2, col3, col4 = st.columns(4)

    # Criar espaços reservados para os gráficos em quatro colunas
    placeholders_col1 = [col1.empty() for _ in range(5)]  # Gráficos 1 a 5
    placeholders_col2 = [col2.empty() for _ in range(4)]  # Gráficos 6 a 9
    placeholders_col3 = [col3.empty() for _ in range(4)]  # Gráficos 10 a 13
    placeholders_col4 = [col4.empty() for _ in range(4)]  # Gráficos 14 a 17

    if st.button("Start"):
        for num_ciclos in range(1, max_ciclo + 1):
            time.sleep(1)  # Simular um delay de 1 segundo para cada ciclo

            # Pivotar os dados para preparar para o modelo
            pivot_x_train = df_tratado.pivot(index=['instancia', 'ciclo_ajustado'], columns='sensor', values='valor').reset_index()
            X_test_pivoted = df_sintetico_concatenado.copy()
            X_test_pivoted = X_test_pivoted[(X_test_pivoted['id'].isin(instancias_para_teste)) & (X_test_pivoted['ciclo_sequencial'] <= num_ciclos)]
            # Salvar a coluna de instância antes de removê-la
            instancias = X_test_pivoted['instancia']
            ids = X_test_pivoted['id']
            
            # Aplicar o scaler separadamente para cada sensor
            scalers = {sensor: MinMaxScaler() for sensor in pivot_x_train.columns if sensor not in ['instancia', 'ciclo_ajustado']}
            # Aplicar o scaler nos dados de treino
            for sensor in scalers:
                if sensor in pivot_x_train.columns:
                    pivot_x_train[sensor] = scalers[sensor].fit_transform(pivot_x_train[[sensor]])
            # # Aplicar o scaler nos dados de teste
            # for sensor in scalers:
            #     if sensor in X_test_pivoted.columns:
            #         X_test_pivoted[sensor] = scalers[sensor].transform(X_test_pivoted[[sensor]])
            #     else:
            #         X_test_pivoted[sensor] = 0
            X_test_pivoted = X_test_pivoted.drop(columns=['instancia', 'ciclo_sequencial', 'id'])
            # Aplicar cada modelo e prever o resultado
            cooler_predictions = model_xgb_cooler.predict(X_test_pivoted)
            valve_predictions = model_xgb_valve.predict(X_test_pivoted)
            leakage_predictions = model_xgb_leakage.predict(X_test_pivoted)
            accumulator_predictions = model_xgb_accumulator.predict(X_test_pivoted)
            def load_from_github(file_url):
                response = requests.get(file_url)
                response.raise_for_status()  # Garantir que a requisição foi bem-sucedida
                return joblib.load(BytesIO(response.content))
            # URLs dos encoders no GitHub
            encoder_cooler_url = base_url + "encoder_cooler.pkl"
            encoder_valve_url = base_url + "encoder_valve.pkl"
            encoder_leakage_url = base_url + "encoder_leakage.pkl"
            encoder_accumulator_url = base_url + "encoder_accumulator.pkl"
            # Carregar encoders do GitHub
            encoder_cooler = load_from_github(encoder_cooler_url)
            encoder_valve = load_from_github(encoder_valve_url)
            encoder_leakage = load_from_github(encoder_leakage_url)
            encoder_accumulator = load_from_github(encoder_accumulator_url)
            cooler_predictions_original = encoder_cooler.inverse_transform(cooler_predictions)
            valve_predictions_original = encoder_valve.inverse_transform(valve_predictions)
            leakage_predictions_original = encoder_leakage.inverse_transform(leakage_predictions)
            accumulator_predictions_original = encoder_accumulator.inverse_transform(accumulator_predictions)
            # Adicionar as previsões ao DataFrame filtrado
            X_test_pivoted_with_results = df_sintetico_concatenado_sem_scaler[(df_sintetico_concatenado_sem_scaler['id'].isin(instancias_para_teste)) & (df_sintetico_concatenado_sem_scaler['ciclo_sequencial'] <= num_ciclos)]
            X_test_pivoted_with_results['cooler_prediction'] = cooler_predictions_original
            X_test_pivoted_with_results['valve_prediction'] = valve_predictions_original
            X_test_pivoted_with_results['leakage_prediction'] = leakage_predictions_original
            X_test_pivoted_with_results['accumulator_prediction'] = accumulator_predictions_original
            
            # # Verificar as primeiras linhas do DataFrame com previsões
            # st.write("X_test_pivoted_with_results:", X_test_pivoted_with_results)
            # Adicionar a coluna de instância de volta ao DataFrame
            X_test_pivoted_with_results['instancia'] = instancias
            X_test_pivoted_with_results['id'] = ids
            # Filtrar os resultados do ciclo selecionado
            resultados_ciclos = X_test_pivoted_with_results[X_test_pivoted_with_results['ciclo_sequencial'] == num_ciclos]
            # Função para converter predições em mensagens
            def get_status_message(prediction, sensor_type):
                if sensor_type == 'cooler':
                    if prediction == 3:
                        return "🔴 Próximo da falha total"
                    elif prediction == 20:
                        return "🟠 Eficiência reduzida"
                    elif prediction == 100:
                        return "🟢 Eficiência total"
                    else:
                        return "⚪ Condição desconhecida"
                elif sensor_type == 'valve':
                    if prediction == 100:
                        return "🟢 Comportamento de comutação ótimo"
                    elif prediction == 90:
                        return "🟡 Pequeno atraso"
                    elif prediction == 80:
                        return "🟠 Atraso severo"
                    elif prediction == 73:
                        return "🔴 Próximo da falha total"
                elif sensor_type == 'leakage':
                    if prediction == 0:
                        return "🟢 Sem vazamento"
                    elif prediction == 1:
                        return "🟡 Vazamento fraco"
                    elif prediction == 2:
                        return "🔴 Vazamento severo"
                elif sensor_type == 'accumulator':
                    if prediction == 130:
                        return "🟢 Pressão ótima"
                    elif prediction == 115:
                        return "🟡 Pressão levemente reduzida"
                    elif prediction == 100:
                        return "🟠 Pressão severamente reduzida"
                    elif prediction == 90:
                        return "🔴 Próximo da falha total"
            # Criar listas para armazenar os dados
            instancia_list = []
            cooler_status_list = []
            valve_status_list = []
            leakage_status_list = []
            accumulator_status_list = []
            # Exibir os resultados para cada instância
            for instancia in instancias_para_teste:
                resultado_instancia = resultados_ciclos[resultados_ciclos['id'] == instancia]
                
                if not resultado_instancia.empty:
                    resultado_cooler = resultado_instancia[['cooler_prediction']].values[0][0]
                    resultado_valve = resultado_instancia[['valve_prediction']].values[0][0]
                    resultado_leakage = resultado_instancia[['leakage_prediction']].values[0][0]
                    resultado_accumulator = resultado_instancia[['accumulator_prediction']].values[0][0]
                    
                    instancia_list.append(instancia)
                    cooler_status_list.append(get_status_message(resultado_cooler, 'cooler'))
                    valve_status_list.append(get_status_message(resultado_valve, 'valve'))
                    leakage_status_list.append(get_status_message(resultado_leakage, 'leakage'))
                    accumulator_status_list.append(get_status_message(resultado_accumulator, 'accumulator'))
            # Criar um DataFrame com os resultados
            resultados_df = pd.DataFrame({
                'Instância': instancia_list,
                'Cooler': cooler_status_list,
                'Resfriador': cooler_status_list,
                'Válvula': valve_status_list,
                'Vazamento': leakage_status_list,
                'Motor': leakage_status_list,
                'Acumulador': accumulator_status_list
            })
        
            # Aplicar estilo para alinhar todas as colunas à esquerda e ajustar o tamanho das colunas
            def align_left(df):
                return df.style.set_properties(**{'text-align': 'left'})
            # Mostrar o DataFrame na tela com estilo aplicado
            st.table(align_left(resultados_df).set_table_styles([{
                'selector': 'th',
                'props': [('text-align', 'left')]
            }, {
                'selector': 'td',
                'props': [('text-align', 'left')]
            }]))
            
            # Filtrar os dados com base no número de ciclos
            X_test_pivoted_with_results = df_sintetico_concatenado_sem_scaler[
                (df_sintetico_concatenado_sem_scaler['id'].isin(instancias_para_teste)) & 
                (df_sintetico_concatenado_sem_scaler['ciclo_sequencial'] <= num_ciclos)
            ]

            # Atualizar os gráficos na coluna 1 (gráficos 1 a 5)
            for idx, sensor in enumerate(lista_sensores[:5]):  # Sensores 1 a 5
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

                # Atualizar o gráfico no espaço reservado correspondente na coluna 1
                placeholders_col1[idx].altair_chart(chart, use_container_width=True)

            # Atualizar os gráficos na coluna 2 (gráficos 6 a 9)
            for idx, sensor in enumerate(lista_sensores[5:9]):  # Sensores 6 a 9
                df_filtrado_sensor = X_test_pivoted_with_results[['ciclo_sequencial', 'id', sensor]].rename(columns={sensor: 'valor', 'ciclo_sequencial': 'ciclo'})

                # Criar um gráfico Altair com interatividade
                chart = alt.Chart(df_filtrado_sensor).mark_line().encode(
                    x='ciclo',
                    y=alt.Y('valor', title=f'Valor ({unidades_sensores[idx + 5]})'),  # Ajustar o índice
                    color=alt.Color('id:N', legend=alt.Legend(title="Instância")),
                    tooltip=['id', 'ciclo', 'valor']
                ).properties(
                    title=f'{nomes_sensores[idx + 5]}'
                ).interactive()  # Permite zoom e pan

                # Atualizar o gráfico no espaço reservado correspondente na coluna 2
                placeholders_col2[idx].altair_chart(chart, use_container_width=True)

            # Atualizar os gráficos na coluna 3 (gráficos 10 a 13)
            for idx, sensor in enumerate(lista_sensores[9:13]):  # Sensores 10 a 13
                df_filtrado_sensor = X_test_pivoted_with_results[['ciclo_sequencial', 'id', sensor]].rename(columns={sensor: 'valor', 'ciclo_sequencial': 'ciclo'})

                # Criar um gráfico Altair com interatividade
                chart = alt.Chart(df_filtrado_sensor).mark_line().encode(
                    x='ciclo',
                    y=alt.Y('valor', title=f'Valor ({unidades_sensores[idx + 9]})'),  # Ajustar o índice
                    color=alt.Color('id:N', legend=alt.Legend(title="Instância")),
                    tooltip=['id', 'ciclo', 'valor']
                ).properties(
                    title=f'{nomes_sensores[idx + 9]}'
                ).interactive()  # Permite zoom e pan

                # Atualizar o gráfico no espaço reservado correspondente na coluna 3
                placeholders_col3[idx].altair_chart(chart, use_container_width=True)

            # Atualizar os gráficos na coluna 4 (gráficos 14 a 17)
            for idx, sensor in enumerate(lista_sensores[13:]):  # Sensores 14 a 17
                df_filtrado_sensor = X_test_pivoted_with_results[['ciclo_sequencial', 'id', sensor]].rename(columns={sensor: 'valor', 'ciclo_sequencial': 'ciclo'})

                # Criar um gráfico Altair com interatividade
                chart = alt.Chart(df_filtrado_sensor).mark_line().encode(
                    x='ciclo',
                    y=alt.Y('valor', title=f'Valor ({unidades_sensores[idx + 13]})'),  # Ajustar o índice
                    color=alt.Color('id:N', legend=alt.Legend(title="Instância")),
                    tooltip=['id', 'ciclo', 'valor']
                ).properties(
                    title=f'{nomes_sensores[idx + 13]}'
                ).interactive()  # Permite zoom e pan

                # Atualizar o gráfico no espaço reservado correspondente na coluna 4
                placeholders_col4[idx].altair_chart(chart, use_container_width=True)
