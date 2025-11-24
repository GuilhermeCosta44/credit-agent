from pydantic import BaseModel, Field, validator
from typing import Literal

class ModelingRequest(BaseModel):
    table_path: str = Field(..., description="Caminho S3 da tabela", example="s3://empresa-credit/tb_full")
    target_column: str = Field(..., description="Nome da coluna target", example="ever30reneg_mob3")
    features_text: str = Field(..., description="Lista de features (texto bruto separado por vírgula ou quebra de linha)")
    metric: Literal['ks2', 'gini', 'auc'] = Field('ks2', description="Métrica principal")

    @validator('features_text')
    def clean_features(cls, v):
        feats = [f.strip() for f in v.replace('\n', ',').split(',') if f.strip()]
        return str(feats)