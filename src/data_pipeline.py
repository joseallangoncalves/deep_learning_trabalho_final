"""
Módulo de Ingestão, Higienização e Pipeline de Pré-processamento Anti-Leakage
Projeto: Voo Regular Ativo (VRA / ANAC) - Disciplina Deep Learning (2026/2)
Prof. Dr. Sérgio Canuto
"""
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_vra_dataset(filepath: str) -> pd.DataFrame:
    """
    Carrega microdados brutos do VRA testando separadores e encodings habituais dos dados abertos.
    """
    separators = [';', ',']
    encodings = ['utf-8', 'latin1', 'cp1252']
    
    for sep in separators:
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc, low_memory=False)
                if 'PartidaPrevista' in df.columns:
                    return df
            except Exception:
                continue
                
    raise ValueError(f"Não foi possível ler o arquivo VRA em '{filepath}'. Verifique formato e encoding.")

def clean_and_prepare_vra(
    df: pd.DataFrame, 
    delay_threshold_min: int = 15,
    top_k_airports: int = 20,
    top_k_airlines: int = 10
) -> pd.DataFrame:
    """
    Higieniza a base do VRA, constrói a variável alvo e extrai variáveis cíclicas.
    Elimina categoricamente colunas geradoras de Data Leakage (ChegadaReal, CodigoJustificativa).
    """
    df_clean = df.copy()
    
    # 1. Filtro de situação de voo (manter apenas voos operados/realizados)
    if 'SituacaoVoo' in df_clean.columns:
        df_clean = df_clean[df_clean['SituacaoVoo'].astype(str).str.upper().isin(['REALIZADO', 'NORMAL'])].copy()
        
    # 2. Conversão segura de datas
    date_cols = ['PartidaPrevista', 'PartidaReal', 'ChegadaPrevista', 'ChegadaReal']
    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            
    df_clean = df_clean.dropna(subset=['PartidaPrevista', 'ChegadaPrevista', 'ChegadaReal']).copy()
    
    # 3. Engenharia da variável alvo (Atraso de Chegada > delay_threshold_min)
    df_clean['AtrasoChegadaMin'] = (df_clean['ChegadaReal'] - df_clean['ChegadaPrevista']).dt.total_seconds() / 60.0
    
    # Exclusão de outliers extremos de digitação física (ex: atrasos < -2h ou > 24h)
    df_clean = df_clean[(df_clean['AtrasoChegadaMin'] >= -120) & (df_clean['AtrasoChegadaMin'] <= 1440)].copy()
    df_clean['target_bin'] = (df_clean['AtrasoChegadaMin'] > delay_threshold_min).astype(int)
    
    # 4. Engenharia de atributos temporais programados
    df_clean['TempoVooPrevistoMin'] = (df_clean['ChegadaPrevista'] - df_clean['PartidaPrevista']).dt.total_seconds() / 60.0
    df_clean = df_clean[df_clean['TempoVooPrevistoMin'] > 0].copy()
    
    # Atributos cíclicos da hora de decolagem programada
    hora_dec = df_clean['PartidaPrevista'].dt.hour + df_clean['PartidaPrevista'].dt.minute / 60.0
    df_clean['hora_sin'] = np.sin(2 * np.pi * hora_dec / 24.0)
    df_clean['hora_cos'] = np.cos(2 * np.pi * hora_dec / 24.0)
    
    # Atributos cíclicos do dia da semana
    dia_sem = df_clean['PartidaPrevista'].dt.dayofweek
    df_clean['dia_semana_sin'] = np.sin(2 * np.pi * dia_sem / 7.0)
    df_clean['dia_semana_cos'] = np.cos(2 * np.pi * dia_sem / 7.0)
    
    # Atributos cíclicos do mês
    mes = df_clean['PartidaPrevista'].dt.month - 1
    df_clean['mes_sin'] = np.sin(2 * np.pi * mes / 12.0)
    df_clean['mes_cos'] = np.cos(2 * np.pi * mes / 12.0)
    
    # 5. Agrupamento de categorias raras (para conter cardinalidade e evitar overfitting)
    if 'ICAOEmpresaAerea' in df_clean.columns:
        top_airlines = df_clean['ICAOEmpresaAerea'].value_counts().nlargest(top_k_airlines).index
        df_clean['ICAOEmpresaAerea'] = df_clean['ICAOEmpresaAerea'].where(df_clean['ICAOEmpresaAerea'].isin(top_airlines), other='OUTROS')
        
    for col in ['ICAOAerodromoOrigem', 'ICAOAerodromoDestino']:
        if col in df_clean.columns:
            top_airports = df_clean[col].value_counts().nlargest(top_k_airports).index
            df_clean[col] = df_clean[col].where(df_clean[col].isin(top_airports), other='OUTROS')
            
    # Criação do identificador de rota agregada
    if 'ICAOAerodromoOrigem' in df_clean.columns and 'ICAOAerodromoDestino' in df_clean.columns:
        df_clean['Rota'] = df_clean['ICAOAerodromoOrigem'] + '_' + df_clean['ICAOAerodromoDestino']
        
    # 6. Ordenação cronológica estrita (prevenir data leakage no particionamento)
    df_clean = df_clean.sort_values('PartidaPrevista').reset_index(drop=True)
    return df_clean

def split_chronological(
    df: pd.DataFrame, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Realiza divisão temporal ordenada estrita (Treino: 70%, Validação: 15%, Teste: 15%).
    Proibido o uso de embaralhamento aleatório (shuffle).
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train:n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_val:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

def build_and_fit_preprocessor(
    train_df: pd.DataFrame, 
    num_cols: List[str], 
    cat_cols: List[str]
) -> ColumnTransformer:
    """
    Ajusta (fit) o ColumnTransformer exclusivamente na partição de Treinamento.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    preprocessor.fit(train_df[num_cols + cat_cols])
    return preprocessor

def transform_splits(
    preprocessor: ColumnTransformer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'target_bin'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica transformação consistente em Treino, Validação e Teste sem vazamento de dados.
    """
    X_train = preprocessor.transform(train_df[feature_cols])
    y_train = train_df[target_col].values.astype(np.float32)
    
    X_val = preprocessor.transform(val_df[feature_cols])
    y_val = val_df[target_col].values.astype(np.float32)
    
    X_test = preprocessor.transform(test_df[feature_cols])
    y_test = test_df[target_col].values.astype(np.float32)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_recurrent_sequences(
    X_features: np.ndarray, 
    y_targets: np.ndarray, 
    time_steps: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Moldagem em tensores 3D (amostras, timesteps, features) para a Frente 3.2 de Redes Recorrentes.
    Preserva a ordem cronológica estrita da série de operações.
    """
    X_seq, y_seq = [], []
    for i in range(len(X_features) - time_steps):
        X_seq.append(X_features[i : i + time_steps])
        y_seq.append(y_targets[i + time_steps])
        
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)
