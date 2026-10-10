# SPEC-01: ESPECIFICAÇÃO DE DADOS, PRÉ-PROCESSAMENTO E PIPELINE ANTI-LEAKAGE
## Trabalho Prático de Deep Learning — Voo Regular Ativo (VRA / ANAC)
### Disciplina: Deep Learning (2026/2) — Prof. Dr. Sérgio Canuto

---

### Identificação da Especificação
- **Identificador:** `SPEC-01`
- **Componente:** Ingestão de Dados, Limpeza, Engenharia de Recursos, Divisão Temporal e Pipeline Sklearn
- **Módulo de Código Alvo:** `src/data_pipeline.py` e `notebooks/01_eda_vra.ipynb` / `notebooks/02_preprocessing_splits.ipynb`
- **Status:** **Aprovado para Implementação**

---

## 1. Origem e Obtenção dos Dados Brutos

Os dados devem ser obtidos do **Portal Brasileiro de Dados Abertos**:
- **Dataset:** Voos e operações aéreas — Voo Regular Ativo (VRA).
- **URL do Portal:** `https://dados.gov.br/dados/conjuntos-dados/dadosabertos-areas-de-atuacao-voos-e-operacoes-aereas-voo-regular-ativo-vra`
- **URL ANAC:** `https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/areas-de-atuacao/voos-e-operacoes-aereas/voo-regular-ativo-vra/62-voo-regular-ativo-vra`
- **Padrão dos Arquivos:** Arquivos mensais em formato CSV (ex: `VRA_2023_01.csv` ou consolidado anual), delimitados por ponto e vírgula (`;`) com codificação `latin1` ou `utf-8`.

---

## 2. Dicionário de Dados e Contrato de Atributos

| Atributo Original | Tipo Esperado | Ação no Pipeline | Justificativa Teórica e Operacional |
| :--- | :--- | :--- | :--- |
| `ICAOEmpresaAerea` | `string` | Categórico (One-Hot) | Concessionárias com perfis operacionais distintos (ex: AZU, GLO, TAM, ONE). Manter top companhias e agrupar raras. |
| `NumeroVoo` | `string` | **Descartado** | Alta cardinalidade e ausência de correlação direta generalizável. |
| `CodigoDI` | `string` | Categórico (One-Hot) | Identificador de voo regular comercial (DI 0, 1, 2...). |
| `CodigoTipoLinha` | `string` | Categórico (One-Hot) | Diferencia linhas Nacionais (`N`), Internacionais (`I`) e Regionais (`R`). |
| `ICAOAerodromoOrigem` | `string` | Categórico (One-Hot) | Aeroporto de decolagem. Reter os top 20 aeródromos e converter demais em `'OUTROS'`. |
| `ICAOAerodromoDestino` | `string` | Categórico (One-Hot) | Aeroporto de pouso. Reter os top 20 aeródromos e converter demais em `'OUTROS'`. |
| `PartidaPrevista` | `datetime` | **Atributo Âncora** | Horário agendado no HOTRAN. Base para ordenação temporal e cálculo de features cíclicas. |
| `PartidaReal` | `datetime` | Uso condicional | Usado para auditoria ou caso o alvo seja atraso de partida. **Descartado** se o objetivo for prever atraso a priori. |
| `ChegadaPrevista` | `datetime` | Feature de tempo | Usado exclusivamente para calcular a duração prevista do voo (`TempoVooPrevistoMin`). |
| `ChegadaReal` | `datetime` | **Cálculo do Alvo** | **Vazamento Direto.** Usado apenas para gerar o rótulo $y$; **removido de $X$**. |
| `SituacaoVoo` | `string` | Filtro de Escopo | Reter linhas com valor `'REALIZADO'`. Linhas `'CANCELADO'` tratadas em escopo apartado. |
| `CodigoJustificativa` | `string` | **Descartado** | **Vazamento Direto.** Preenchido somente após o registro do atraso pela companhia aérea. |

---

## 3. Regras de Blindagem Contra Vazamento de Dados (Anti-Leakage)

Conforme os slides de **Parametrização** do Prof. Sérgio Canuto:

1. **Ordenação Temporal Estrita:**
   ```python
   df = df.sort_values('PartidaPrevista').reset_index(drop=True)
   ```
   É terminantemente proibido o uso de `train_test_split` com `shuffle=True`. O modelo deve aprender no passado e prever no futuro.

2. **Divisão das Partições:**
   - **Treino ($D_{\text{treino}}$):** $70\%$ iniciais cronológicos.
   - **Validação ($D_{\text{val}}$):** $15\%$ subsequentes cronológicos.
   - **Teste ($D_{\text{teste}}$):** $15\%$ finais cronológicos.

3. **Isolamento de Ajuste Estatístico (`fit` vs. `transform`):**
   - Médias ($\mu$), desvios-padrão ($\sigma$), mínimos, máximos e vocabulários de categorias devem ser aprendidos **exclusivamente no conjunto de treinamento**.
   - Validação e teste recebem exclusivamente `.transform()`.

---

## 4. Engenharia de Recursos (Feature Engineering)

### 4.1 Extração de Variáveis Cíclicas
Horários e calendários possuem natureza circular ($23:59 \to 00:00$). Para permitir que as redes neurais interpretem essa continuidade sem descontinuidade métrica:

$$\text{Hora}_{\sin} = \sin\left(\frac{2\pi \cdot \text{HoraDecimal}}{24}\right), \quad \text{Hora}_{\cos} = \cos\left(\frac{2\pi \cdot \text{HoraDecimal}}{24}\right)$$

$$\text{DiaSemana}_{\sin} = \sin\left(\frac{2\pi \cdot \text{DiaSemana}}{7}\right), \quad \text{DiaSemana}_{\cos} = \cos\left(\frac{2\pi \cdot \text{DiaSemana}}{7}\right)$$

$$\text{Mes}_{\sin} = \sin\left(\frac{2\pi \cdot (\text{Mes} - 1)}{12}\right), \quad \text{Mes}_{\cos} = \cos\left(\frac{2\pi \cdot (\text{Mes} - 1)}{12}\right)$$

### 4.2 Duração Programada da Rota
$$\text{TempoVooPrevistoMin} = \frac{\text{ChegadaPrevista} - \text{PartidaPrevista}}{60 \text{ segundos}}$$

### 4.3 Criação do Rótulo Alvo
$$\text{AtrasoChegadaMin} = \frac{\text{ChegadaReal} - \text{ChegadaPrevista}}{60 \text{ segundos}}$$
$$y = \begin{cases} 1, & \text{se } \text{AtrasoChegadaMin} > 15 \text{ minutos} \\ 0, & \text{caso contrário} \end{cases}$$

---

## 5. Implementação de Referência do Módulo `src/data_pipeline.py`

```python
"""
Módulo de Ingestão e Processamento de Dados VRA / ANAC
Em estrito alinhamento às diretrizes do Prof. Sérgio Canuto.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_vra_dataset(filepath: str) -> pd.DataFrame:
    for sep in [';', ',']:
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc, low_memory=False)
                if 'PartidaPrevista' in df.columns:
                    return df
            except Exception:
                continue
    raise ValueError(f"Não foi possível decodificar o arquivo {filepath}")

def clean_and_prepare_vra(df: pd.DataFrame, delay_threshold_min: int = 15) -> pd.DataFrame:
    # 1. Filtro de voos realizados
    if 'SituacaoVoo' in df.columns:
        df = df[df['SituacaoVoo'].str.upper().isin(['REALIZADO', 'NORMAL'])].copy()
    
    # 2. Conversão segura de datas
    date_cols = ['PartidaPrevista', 'PartidaReal', 'ChegadaPrevista', 'ChegadaReal']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    df = df.dropna(subset=['PartidaPrevista', 'ChegadaPrevista', 'ChegadaReal']).copy()
    
    # 3. Engenharia da variável alvo (sem leakage)
    df['AtrasoChegadaMin'] = (df['ChegadaReal'] - df['ChegadaPrevista']).dt.total_seconds() / 60.0
    # Limpeza de outliers absurdos (ex: inconsistências de digitação > 24h ou < -5h)
    df = df[(df['AtrasoChegadaMin'] >= -120) & (df['AtrasoChegadaMin'] <= 1440)].copy()
    df['target_bin'] = (df['AtrasoChegadaMin'] > delay_threshold_min).astype(int)
    
    # 4. Features contínuas e cíclicas
    df['TempoVooPrevistoMin'] = (df['ChegadaPrevista'] - df['PartidaPrevista']).dt.total_seconds() / 60.0
    df = df[df['TempoVooPrevistoMin'] > 0].copy()
    
    hora_dec = df['PartidaPrevista'].dt.hour + df['PartidaPrevista'].dt.minute / 60.0
    df['hora_sin'] = np.sin(2 * np.pi * hora_dec / 24.0)
    df['hora_cos'] = np.cos(2 * np.pi * hora_dec / 24.0)
    
    dia_sem = df['PartidaPrevista'].dt.dayofweek
    df['dia_semana_sin'] = np.sin(2 * np.pi * dia_sem / 7.0)
    df['dia_semana_cos'] = np.cos(2 * np.pi * dia_sem / 7.0)
    
    mes = df['PartidaPrevista'].dt.month - 1
    df['mes_sin'] = np.sin(2 * np.pi * mes / 12.0)
    df['mes_cos'] = np.cos(2 * np.pi * mes / 12.0)
    
    # 5. Ordenação temporal obrigatória
    df = df.sort_values('PartidaPrevista').reset_index(drop=True)
    return df

def split_chronological(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_val].copy()
    test_df = df.iloc[n_val:].copy()
    return train_df, val_df, test_df

def build_and_fit_pipeline(train_df: pd.DataFrame, num_cols: list, cat_cols: list):
    # Tratar categorias raras no treino
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    preprocessor.fit(train_df[num_cols + cat_cols])
    return preprocessor
```

---

## 6. Critérios de Aceite da SPEC-01
- [ ] O pipeline executa `fit` estritamente na partição `train_df`.
- [ ] Atributos pós-fato (`ChegadaReal`, `CodigoJustificativa`) estão 100% ausentes do conjunto preditor $X$.
- [ ] Nenhuma sobreposição temporal entre conjuntos de Treino, Validação e Teste.
- [ ] Verificação de dimensionalidade e tipo com asserção: `assert X_train.shape[0] == len(train_df)`.
