from pydantic import BaseModel, Field, validator
from typing import Literal

class ModelingRequest(BaseModel):
    table_path: str = Field(..., description="Caminho S3/DBFS da tabela de dados")
    target_column: str = Field(..., description="Nome da coluna target")
    features_text: str = Field(..., description="Lista de features")
    metric: Literal['ks2', 'gini', 'auc'] = Field('ks2', description="Métrica principal")
    run_path: str = Field(..., description="Caminho do notebook de funções no Databricks (para o comando %run)")

    @validator('features_text')
    def clean_features(cls, v):
        feats = [f.strip() for f in v.replace('\n', ',').split(',') if f.strip()]
        return str(feats)