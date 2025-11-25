import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from app.models import ModelingRequest
from app.services.knowledge_loader import load_reference_cells

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
    print("1. Carregando funções da Base de Conhecimento...")
    static_code_cells = load_reference_cells()
    
   
    model = genai.GenerativeModel('models/gemini-2.0-flash-lite-preview-02-05')
    
    prompt = f"""
    Você é um Arquiteto de ML. Sua tarefa é escrever APENAS O CÓDIGO PYTHON da célula de execução de um notebook.
    
    CONTEXTO:
    Todas as funções de library (calculate_metrics, features_binning_process, fs_rfe_lgbm, main, etc) JÁ FORAM DEFINIDAS em células anteriores. NÃO AS REDEFINA.
    
    SEU TRABALHO:
    Escreva um script Python que use essas funções para rodar o pipeline com os parâmetros abaixo.
    
    PARÂMETROS DO USUÁRIO:
    - Tabela: "{params.table_path}"
    - Target: "{params.target_column}"
    - Features: {params.features_text}
    - Métrica: "{params.metric}"
    
    O CÓDIGO DEVE CONTER EXATAMENTE ESTA LÓGICA:
    1. Definir variáveis globais (TABLE_PATH, TARGET, FEATURES, METRIC).
    2. Ler os dados: df = spark.read.parquet(TABLE_PATH)
    3. Filtrar base de desenvolvimento (DEV) e OOT. Se não tiver coluna 'dev', criar split aleatório.
    4. Chamar a função: features_binning_process(...)
    5. Chamar a função: fs_iv(...)
    6. Chamar a função: autoEliminateMulticollinearityHybrid(...)
    7. Chamar a função: main(...) para otimização.
    8. Exibir os melhores resultados.
    
    SAÍDA:
    Retorne APENAS o código Python puro. Sem markdown, sem json, sem explicações.
    """

    print("2. Solicitando ao Gemini apenas a lógica de execução...")
    try:
        response = model.generate_content(prompt)
        execution_code = response.text.replace("```python", "").replace("```", "").strip()
        
      
        notebook_cells = [
            create_cell(f"# Projeto Automático: {params.target_column}\nGerado por IA.", "markdown"),
            create_cell("%pip install --upgrade optbinning tqdm mlflow==2.11.2 shap optuna optuna-integration xgboost catboost scikit-learn\ndbutils.library.restartPython()", "code")
        ]
        
        notebook_cells.append(create_cell("# --- BIBLIOTECA DE FUNÇÕES (Carregadas da Base de Conhecimento) ---", "markdown"))
        for code in static_code_cells:
            notebook_cells.append(create_cell(code, "code"))
            
        notebook_cells.append(create_cell("# --- PIPELINE DE EXECUÇÃO ---", "markdown"))
        notebook_cells.append(create_cell(execution_code, "code"))

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

    except Exception as e:
        raise ValueError(f"Erro na geração: {str(e)}")