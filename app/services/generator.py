import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
from app.models import ModelingRequest
from app.services.knowledge_loader import load_reference_cells
from google.api_core import exceptions

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def create_cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

def generate_notebook_json(params: ModelingRequest) -> str:
    print("1. Lendo 'Manual de Instruções' das funções...")
    # O Agente lê o arquivo local para APRENDER a usar as funções, não para copiar.
    reference_context = load_reference_cells()
    
    # Modelo Lite (Rápido e eficiente)
    model_name = 'models/gemini-2.0-flash-lite-preview-02-05'
    model = genai.GenerativeModel(model_name)
    
    # Prompt focado em ANÁLISE e EXECUÇÃO
    prompt = f"""
    ATUE COMO: Lead Data Scientist Especialista em Crédito.
    OBJETIVO: Escrever o código de EXECUÇÃO E ANÁLISE para um notebook Databricks.
    
    ### CENÁRIO
    Todas as funções complexas (feature selection, binning, modelagem, métricas) JÁ EXISTEM e serão carregadas na memória via comando mágico %run.
    NÃO redefina as funções. O seu trabalho é USÁ-LAS para criar uma análise completa.

    ### SUA BIBLIOTECA DE FERRAMENTAS (Apenas para consulta de uso):
    {reference_context}
    (Consulte acima: nomes dos parâmetros e o que cada função retorna para usar corretamente).

    ### DADOS DO PROJETO ATUAL:
    - Tabela Input: "{params.table_path}"
    - Target: "{params.target_column}"
    - Features: {params.features_text}
    - Métrica de Sucesso: "{params.metric}"

    ### TAREFA: Gerar o Script de Execução (Python)
    Escreva um script Python longo e detalhado que faça o seguinte fluxo:

    1. Configuração: Defina variáveis globais (TABLE_PATH, TARGET, FEATURES, METRIC).
    2. Leitura: Carregue os dados (spark.read.parquet). Se não houver coluna 'dev', crie um split aleatório.
    3. Feature Engineering (Com Análise):
       - Chame `features_binning_process`.
       - Use `display()` nos dataframes retornados para mostrar a qualidade dos bins.
       - Chame `fs_iv` para calcular Information Value. Dê `display()` na tabela de IV.
       - Chame `autoEliminateMulticollinearityHybrid`. Mostre a matriz de correlação final.
    4. Modelagem (AutoML):
       - Chame a função `main` (que roda o Optuna). Passe os dataframes processados.
       - Dê `display()` no dataframe de resultados do Optuna, ordenado pela métrica escolhida.
    5. Visualização Final:
       - Chame `plot_best_model_express` para gerar os gráficos de performance (KS/AUC).
       - Chame `graph_feature_importance` e `graph_shap_value` usando o ID do melhor modelo (pegue do MLflow ou do retorno da função main).

    ### REGRAS DE OURO:
    - Use `display(df)` do Databricks para TODAS as tabelas intermediárias.
    - Adicione comentários explicativos no código (# Explicando o insight).
    - NÃO inclua blocos markdown (```python), apenas o código puro.
    """

    print("2. Solicitando ao Gemini o Pipeline Analítico...")
    
    execution_content = ""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            execution_content = response.text.replace("```python", "").replace("```", "").strip()
            break
        except Exception as e:
            print(f"Tentativa {attempt+1} falhou: {e}")
            if attempt == max_retries - 1: raise ValueError(f"Erro na IA: {e}")
            time.sleep(5)

    notebook_cells = []

    notebook_cells.append(create_cell(f"# 📊 Relatório de Modelagem Automática: {params.target_column}\nNotebook gerado por IA. Foco em análise e interpretabilidade.", "markdown"))

    notebook_cells.append(create_cell("# Instalação das bibliotecas necessárias\n%pip install --upgrade optbinning tqdm mlflow==2.11.2 shap optuna optuna-integration xgboost catboost scikit-learn\ndbutils.library.restartPython()", "code"))

    notebook_cells.append(create_cell(f"# 🧠 Carregando Base de Conhecimento (Funções de Crédito)\n# Certifique-se que este caminho existe no seu Workspace\n%run \"{params.run_path}\"", "code"))

    notebook_cells.append(create_cell("# Imports locais para análise\nimport pandas as pd\nimport numpy as np\nfrom pyspark.sql import functions as f\nimport mlflow", "code"))

    notebook_cells.append(create_cell("## 🚀 Execução do Pipeline Analítico", "markdown"))
    notebook_cells.append(create_cell(execution_content, "code"))

    notebook_cells.append(create_cell("# Fim da Execução. Verifique os artefatos no MLflow.", "markdown"))

    notebook_json = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return json.dumps(notebook_json, indent=2)