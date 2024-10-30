import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import requests
from datetime import datetime

def verificar_data_horario():

    # Pega a data e hora atual
    data_hora = datetime.now()
    
    # Verifica se é final de semana (sábado ou domingo)
    final_de_semana_real = 1 if data_hora.weekday() >= 5 else 0  # Sábado (5) e domingo (6)
    
    # Verifica se está no horário comercial (08:00 às 17:00)
    horario_comercial_real = 1 if 8 <= data_hora.hour < 17 else 0

    return final_de_semana_real, horario_comercial_real

# Exemplo de uso
final_de_semana_real, horario_comercial_real = verificar_data_horario()

# Defina a cidade e a sua chave da API
cidade = "Rio de Janeiro"  # Altere para a cidade desejada
api_key = "af14a5d98cd1cc1fc38ce560697d2727"
url = "http://api.openweathermap.org/data/2.5/weather?q=Rio%20de%20Janeiro&appid=af14a5d98cd1cc1fc38ce560697d2727&lang=pt&units=metric"


# Faz a solicitação para a API
resposta = requests.get(url)

dados_clima = resposta.json()
pressao_real = dados_clima['main']['pressure']
umidade_real = dados_clima['main']['humidity']
temperatura_real = dados_clima['main']['temp']



# Função para definir o CSS personalizado
def set_css():
    st.markdown(
        """
        <style>
        .custom-title {
            font-size: 30px !important;
        }
        .custom-subtitle {
            font-size: 17px !important;
        }
        </style>
        """, unsafe_allow_html=True
    )

# Adicionar o CSS personalizado para o tamanho da fonte
set_css()

#pathProd = ''
pathProd = 'USP_TCC_prediction_bms_Streamlit/'

# Função para carregar o modelo, tentando ambos os formatos (.pkl e .h5)
import os

def carregar_modelo(joblib_path, keras_path):
    """
    Função para carregar o modelo. Verifica se o arquivo existe antes de tentar carregá-lo.
    """
    try:
        # Verifica se o arquivo Keras (.h5) existe
        if os.path.exists(keras_path):
            return load_model(keras_path)
        # Verifica se o arquivo Joblib (.pkl) existe
        elif os.path.exists(joblib_path):
            return joblib.load(joblib_path)
        else:
            st.error(f"Modelo não encontrado nos caminhos: {keras_path} ou {joblib_path}")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função para prever valores para um chiller específico

def calcular_previsoes_ahu(model_staMedia_ahu0302, scaler_staMedia_ahu0302, model_ti_ahu0302, scaler_ti_ahu0302):
    
    input_data_staMedia = np.array([[pressao, temperatura, previsaoTI, vagAhu]])
    input_data_scaled_staMedia = scaler_staMedia_ahu0302.transform(input_data_staMedia)
    previsaostaMedia = model_staMedia_ahu0302.predict(input_data_scaled_staMedia).flatten()


    return previsaostaMedia, previsaoTI


def calcular_previsoes(scaler_corrente, scaler_deltaAC, scaler_Ligados, scaler_TR, scaler_VAG, scaler_KWH, scaler_torre3, 
                       model_corrente, model_deltaAC, model_Ligados, model_TR, model_VAG, model_KWH, model_torre3, 
                       ur_temp_saida):  # Incluindo ur_temp_saida como último parâmetro

    # Use o valor `ur_temp_saida` normalmente na função
    input_data_Ligados = np.array([[pressao, temperatura, umidade, FimDeSemana, HorarioComercial]])
    input_data_scaled_Ligados = scaler_Ligados.transform(input_data_Ligados)
    previsaoLigados = model_Ligados.predict(input_data_scaled_Ligados).flatten()

    input_data_VAG = np.array([[pressao, temperatura, umidade, FimDeSemana, HorarioComercial, previsaoLigados[0]]])
    input_data_scaled_VAG = scaler_VAG.transform(input_data_VAG)
    previsaoVAG = model_VAG.predict(input_data_scaled_VAG).flatten()

    input_data_deltaAC = np.array([[pressao, temperatura, umidade, ur_temp_saida, previsaoVAG[0], previsaoLigados[0]]])
    input_data_scaled_deltaAC = scaler_deltaAC.transform(input_data_deltaAC)
    previsaodeltaAC = model_deltaAC.predict(input_data_scaled_deltaAC).flatten()

    input_data_torre3 = np.array([[pressao, temperatura, umidade, previsaodeltaAC[0], previsaoVAG[0]]])
    input_data_scaled_torre3 = scaler_torre3.transform(input_data_torre3)
    previsaoTorre3 = model_torre3.predict(input_data_scaled_torre3).flatten()

    input_data_TR = np.array([[pressao, temperatura, umidade, previsaodeltaAC[0], previsaoVAG[0], ur_temp_saida, FimDeSemana, HorarioComercial, previsaoLigados[0], previsaoTorre3[0]]])
    input_data_scaled_TR = scaler_TR.transform(input_data_TR)
    previsaoTR = model_TR.predict(input_data_scaled_TR).flatten()

    input_data_KWH = np.array([[pressao, temperatura, umidade, previsaodeltaAC[0], previsaoTR[0], ur_temp_saida, previsaoVAG[0], previsaoTorre3[0], previsaoLigados[0]]])
    input_data_scaled_KWH = scaler_KWH.transform(input_data_KWH)
    previsaoKWH = model_KWH.predict(input_data_scaled_KWH).flatten()

    input_data_corrente = np.array([[pressao, temperatura, umidade, ur_temp_saida, previsaoTR[0], previsaodeltaAC[0], previsaoVAG[0], previsaoLigados[0], previsaoKWH[0], previsaoTorre3[0]]])
    input_data_scaled_corrente = scaler_corrente.transform(input_data_corrente)
    previsaoCorrente = model_corrente.predict(input_data_scaled_corrente).flatten()

    return previsaoCorrente, previsaoVAG, previsaoLigados, previsaodeltaAC, previsaoTR, previsaoKWH, previsaoTorre3

# Carregar o modelo e o scaler Chiller 1
model_corrente_chiller1 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller1/ur_correnteMotor/model.pkl', f'{pathProd}ModelsDeploy/chiller1/ur_correnteMotor/model.h5')
scaler_corrente_chiller1 = joblib.load(f'{pathProd}ModelsDeploy/chiller1/ur_correnteMotor/scaler.pkl')

model_deltaAC_chiller1 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller1/delta_AC/model.pkl', f'{pathProd}ModelsDeploy/chiller1/delta_AC/model.h5')
scaler_deltaAC_chiller1 = joblib.load(f'{pathProd}ModelsDeploy/chiller1/delta_AC/scaler.pkl')

model_Ligados_chiller1 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller1/Fancoil_ligado_%/model.pkl', f'{pathProd}ModelsDeploy/chiller1/Fancoil_ligado_%/model.h5')
scaler_Ligados_chiller1 = joblib.load(f'{pathProd}ModelsDeploy/chiller1/Fancoil_ligado_%/scaler.pkl')

model_TR_chiller1 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller1/TR/model.pkl', f'{pathProd}ModelsDeploy/chiller1/TR/model.h5')
scaler_TR_chiller1 = joblib.load(f'{pathProd}ModelsDeploy/chiller1/TR/scaler.pkl')

model_VAG_chiller1 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller1/VAG_Aberta_%/model.pkl', f'{pathProd}ModelsDeploy/chiller1/VAG_Aberta_%/model.h5')
scaler_VAG_chiller1 = joblib.load(f'{pathProd}ModelsDeploy/chiller1/VAG_Aberta_%/scaler.pkl')

model_KWH_chiller1 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller1/ur_kwh/model.pkl', f'{pathProd}ModelsDeploy/chiller1/ur_kwh/model.h5')
scaler_KWH_chiller1 = joblib.load(f'{pathProd}ModelsDeploy/chiller1/ur_kwh/scaler.pkl')

model_torre3_chiller1 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller1/Torre_3/model.pkl', f'{pathProd}ModelsDeploy/chiller1/Torre_3/model.h5')
scaler_torre3_chiller1 = joblib.load(f'{pathProd}ModelsDeploy/chiller1/Torre_3/scaler.pkl')

# Carregar o modelo e o scaler Chiller 2
model_corrente_chiller2 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller2/ur_correnteMotor/model.pkl', f'{pathProd}ModelsDeploy/chiller2/ur_correnteMotor/model.h5')
scaler_corrente_chiller2 = joblib.load(f'{pathProd}ModelsDeploy/chiller2/ur_correnteMotor/scaler.pkl')

model_deltaAC_chiller2 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller2/delta_AC/model.pkl', f'{pathProd}ModelsDeploy/chiller2/delta_AC/model.h5')
scaler_deltaAC_chiller2 = joblib.load(f'{pathProd}ModelsDeploy/chiller2/delta_AC/scaler.pkl')

model_Ligados_chiller2 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller2/Fancoil_ligado_%/model.pkl', f'{pathProd}ModelsDeploy/chiller2/Fancoil_ligado_%/model.h5')
scaler_Ligados_chiller2 = joblib.load(f'{pathProd}ModelsDeploy/chiller2/Fancoil_ligado_%/scaler.pkl')

model_TR_chiller2 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller2/TR/model.pkl', f'{pathProd}ModelsDeploy/chiller2/TR/model.h5')
scaler_TR_chiller2 = joblib.load(f'{pathProd}ModelsDeploy/chiller2/TR/scaler.pkl')

model_VAG_chiller2 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller2/VAG_Aberta_%/model.pkl', f'{pathProd}ModelsDeploy/chiller2/VAG_Aberta_%/model.h5')
scaler_VAG_chiller2 = joblib.load(f'{pathProd}ModelsDeploy/chiller2/VAG_Aberta_%/scaler.pkl')

model_KWH_chiller2 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller2/ur_kwh/model.pkl', f'{pathProd}ModelsDeploy/chiller2/ur_kwh/model.h5')
scaler_KWH_chiller2 = joblib.load(f'{pathProd}ModelsDeploy/chiller2/ur_kwh/scaler.pkl')

model_torre3_chiller2 = carregar_modelo(f'{pathProd}ModelsDeploy/chiller2/Torre_3/model.pkl', f'{pathProd}ModelsDeploy/chiller2/Torre_3/model.h5')
scaler_torre3_chiller2 = joblib.load(f'{pathProd}ModelsDeploy/chiller2/Torre_3/scaler.pkl')

model_staMedia_ahu0302 = carregar_modelo(f'{pathProd}ModelsDeploy/AHU-03-02/STA_media/model.pkl', f'{pathProd}ModelsDeploy/AHU-03-02/STA_media/model.h5')
scaler_staMedia_ahu0302 = joblib.load(f'{pathProd}ModelsDeploy/AHU-03-02/STA_media/scaler.pkl')

model_ti_ahu0302 = carregar_modelo(f'{pathProd}ModelsDeploy/AHU-03-02/TI/model.pkl', f'{pathProd}ModelsDeploy/AHU-03-02/TI/model.h5')
scaler_ti_ahu0302 = joblib.load(f'{pathProd}ModelsDeploy/AHU-03-02/TI/scaler.pkl')






# Ajustar o layout: sliders na lateral
st.sidebar.title("Parâmetros de Entrada")

# Sliders para os parâmetros de entrada
pressao = st.sidebar.text_input('Pressão (mB)', value=str(pressao_real), key="pressao_input")
temperatura = st.sidebar.text_input('Temperatura (°C)', value=str(temperatura_real), key="temperatura_input")
umidade = st.sidebar.text_input('Umidade (%)', value=str(umidade_real), key="umidade_input")
FimDeSemana = st.sidebar.selectbox('Fim de Semana', [0, 1], index=final_de_semana_real, key="fim_de_semana_input")
HorarioComercial = st.sidebar.selectbox('Horário Comercial', [0, 1], index=horario_comercial_real, key="horario_comercial_input")

# Adicionar abas para Chiller 1, Chiller 2, Comparativo e AHU-03-02
tab1, tab2, tab3, tab4 = st.tabs(["Chiller 1", "Chiller 2", "Comparativo", "AHU-03-02"])

# Aba Chiller 1
with tab1:
    st.markdown('<h1 class="custom-title">Previsões de Desempenho - Chiller 1</h1>', unsafe_allow_html=True)
    ur_temp_saida = st.text_input('Temperatura de Saída (°C)', value=9.5, key="ur_temp_saida_chiller1")

   
    previsaoCorrente, previsaoVAG, previsaoLigados, previsaodeltaAC, previsaoTR, previsaoKWH, previsaoTorre3 = calcular_previsoes(
        scaler_corrente_chiller1, scaler_deltaAC_chiller1, scaler_Ligados_chiller1, 
        scaler_TR_chiller1, scaler_VAG_chiller1, scaler_KWH_chiller1, scaler_torre3_chiller1,
        model_corrente_chiller1, model_deltaAC_chiller1, model_Ligados_chiller1, 
        model_TR_chiller1, model_VAG_chiller1, model_KWH_chiller1, model_torre3_chiller1,
        ur_temp_saida  # Passar o valor atualizado de `ur_temp_saida`
    )

    resultados_1 = pd.DataFrame({
        'Parâmetro': ['Corrente (%)', 'VAG (%)', 'Fancoils Ligados (%)', 'Delta AC (°C)', 'TR', 'KWH', 'Torre Hz'],
        'Previsão': [
            f'{previsaoCorrente[0]:.2f}',
            f'{previsaoVAG[0]:.2f}', 
            f'{previsaoLigados[0]:.2f}', 
            f'{previsaodeltaAC[0]:.2f}', 
            f'{previsaoTR[0]:.2f}',
            f'{previsaoKWH[0]:.2f}',  
            f'{previsaoTorre3[0]:.2f}'  
        ]
    })

    st.table(resultados_1.style.set_properties(**{'text-align': 'center'}))

# Aba Chiller 2
with tab2:
    st.markdown('<h1 class="custom-title">Previsões de Desempenho - Chiller 2</h1>', unsafe_allow_html=True)
    ur_temp_saida = st.text_input('Temperatura de Saída (°C)', value=9.5, key="ur_temp_saida_chiller2")

    previsaoCorrente_2, previsaoVAG_2, previsaoLigados_2, previsaodeltaAC_2, previsaoTR_2, previsaoKWH_2, previsaoTorre3_2 = calcular_previsoes(
        scaler_corrente_chiller2, scaler_deltaAC_chiller2, scaler_Ligados_chiller2, 
        scaler_TR_chiller2, scaler_VAG_chiller2, scaler_KWH_chiller2, scaler_torre3_chiller2,
        model_corrente_chiller2, model_deltaAC_chiller2, model_Ligados_chiller2, 
        model_TR_chiller2, model_VAG_chiller2, model_KWH_chiller2, model_torre3_chiller2,
        ur_temp_saida  # Passar o valor atualizado de `ur_temp_saida`
    )

    resultados_2 = pd.DataFrame({
        'Parâmetro': ['Corrente (%)', 'VAG (%)', 'Fancoils Ligados (%)', 'Delta AC (°C)', 'TR', 'KWH', 'Torre Hz'],
        'Previsão': [
            f'{previsaoCorrente_2[0]:.2f}',
            f'{previsaoVAG_2[0]:.2f}', 
            f'{previsaoLigados_2[0]:.2f}', 
            f'{previsaodeltaAC_2[0]:.2f}', 
            f'{previsaoTR_2[0]:.2f}',
            f'{previsaoKWH_2[0]:.2f}',  
            f'{previsaoTorre3_2[0]:.2f}'  
        ]
    })

    st.table(resultados_2.style.set_properties(**{'text-align': 'center'}))

# Aba Comparativo
with tab3:
    st.markdown('<h1 class="custom-title">Comparativo</h1>', unsafe_allow_html=True)

    # Valores de previsão para o gráfico
    parametros = ['Corrente (%)', 'VAG (%)', 'Fancoils Ligados (%)', 'Delta AC (°C)', 'TR', 'KWH']
    
    chiller_1_valores = [float(previsaoCorrente[0]), float(previsaoVAG[0]), float(previsaoLigados[0]), float(previsaodeltaAC[0]), float(previsaoTR[0]), float(previsaoKWH[0])]
    chiller_2_valores = [float(previsaoCorrente_2[0]), float(previsaoVAG_2[0]), float(previsaoLigados_2[0]), float(previsaodeltaAC_2[0]), float(previsaoTR_2[0]), float(previsaoKWH_2[0])]

    # Criar gráficos interativos para cada parâmetro
    for i, parametro in enumerate(parametros):
        fig = go.Figure()

        # Adicionar barras de Chiller 1 e Chiller 2 com os valores nas colunas
        fig.add_trace(go.Bar(
            x=['Chiller 1'], 
            y=[chiller_1_valores[i]], 
            name='Chiller 1',
            text=[f'{chiller_1_valores[i]:.2f}'],  
            textposition='inside'  
        ))

        fig.add_trace(go.Bar(
            x=['Chiller 2'], 
            y=[chiller_2_valores[i]], 
            name='Chiller 2',
            text=[f'{chiller_2_valores[i]:.2f}'],  
            textposition='inside'  
        ))

        # Configurações do gráfico
        fig.update_layout(
            title=f'Comparação de {parametro} entre Chiller 1 e Chiller 2',
            xaxis_title='Chiller',
            yaxis_title=parametro,
            barmode='group',
            uniformtext_minsize=8,  
            uniformtext_mode='hide'  
        )

        # Exibir o gráfico no Streamlit
        st.plotly_chart(fig)

# Aba AHU-03-02
with tab4:
    st.markdown('<h1 class="custom-title">Previsões de Desempenho - AHU-03-02</h1>', unsafe_allow_html=True)

    # Número de entrada para VAG AHU (%) apenas na aba AHU-03-02
    vagAhu = st.number_input('VAG AHU (%)', min_value=0.0, max_value=100.0, value=76.0, key="vagAhu_ahu")
    previsaoTI = st.number_input('Insuflamento (°C)', min_value=10.0, max_value=18.0, value=15.0, key="previsaoTI_ahu")

    previsaostaMedia, previsaoTI = calcular_previsoes_ahu(model_staMedia_ahu0302, scaler_staMedia_ahu0302, model_ti_ahu0302, scaler_ti_ahu0302)

    resultados_3 = pd.DataFrame({
        'Parâmetro': ['Temperatura Média ambiente'],
        'Previsão': [
            f'{previsaostaMedia[0]:.2f}'
        ]
    })

    st.table(resultados_3.style.set_properties(**{'text-align': 'center'}))
