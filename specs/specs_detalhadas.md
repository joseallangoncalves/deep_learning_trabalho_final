# ESPECIFICAÇÕES TÉCNICAS DETALHADAS (SPECS v1.0.0)
## Trabalho Prático de Deep Learning — VRA / ANAC (Dados Abertos)
### Disciplina: Deep Learning (2026/2) — Prof. Dr. Sérgio Canuto (IFG)

---

### Controle de Versão do Documento de Especificação
| Versão | Data | Responsável | Descrição da Especificação | Status |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0.0** | 10/10/2026 | Equipe de P&D | Especificação formal completa de dados, arquiteturas Keras, busca Optuna, pipeline de avaliação e módulos `src/`. | **Homologado para Codificação** |

---

## 1. Escopo e Princípios Arquiteturais

Este documento define as especificações formais de engenharia de software e experimentação científica para implementação do trabalho prático de Deep Learning, em estrita aderência ao:
1. **Edital do Trabalho:** `Trabalho - Disciplina Deep Learning (2).pdf`.
2. **Plano de Implementação Versionado:** [plano_implementacao_versionado.md](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/specs/plano_implementacao_versionado.md).
3. **Guia Metodológico VRA:** [deep_learning_trabalho.html](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/documentos/deep_learning_trabalho.html).
4. **Base RAG de Aulas:** 8 apresentações de slides do Prof. Sérgio Canuto.

---

## 2. SPEC-01: Contrato de Dados e Engenharia de Features (Data Contract)

### 2.1 Esquema dos Microdados Brutos do VRA
A ingestão processa arquivos CSV delimitados por ponto e vírgula (`;`) ou vírgula (`,`) obtidos do Portal Brasileiro de Dados Abertos da ANAC.

| Atributo Original | Tipo Pandas | Descrição de Domínio | Política de Tratamento Inicial |
| :--- | :--- | :--- | :--- |
| `ICAOEmpresaAerea` | `string / category` | Código ICAO da empresa (ex: `AZU`, `GLO`, `TAM`) | Manter principais; agregar categorias raras como `'OUTROS'`. |
| `NumeroVoo` | `string / int` | Número identificador do voo no HOTRAN | Excluir como preditor (alta cardinalidade, risco de overfitting). |
| `CodigoDI` | `string / category` | Código do tipo de voo (0: Regular, etc.) | One-Hot Encoding ou filtro de voos comerciais regulares. |
| `CodigoTipoLinha` | `string / category` | Nacional (`N`), Internacional (`I`), Regional (`R`) | One-Hot Encoding. |
| `ICAOAerodromoOrigem` | `string / category` | Código ICAO de partida (ex: `SBBR`, `SBGR`, `SBCF`) | Manter os $K$ maiores aeroportos nacionais; demais como `'OUTROS'`. |
| `ICAOAerodromoDestino` | `string / category` | Código ICAO de destino (ex: `SBSP`, `SBRJ`, `SBKP`) | Manter os $K$ maiores aeroportos nacionais; demais como `'OUTROS'`. |
| `PartidaPrevista` | `datetime64[ns]` | Horário programado da decolagem | Atributo base temporal para ordenação e extração cíclica. |
| `PartidaReal` | `datetime64[ns]` | Horário efetivo da decolagem | Utilizado apenas para cálculo de atraso de partida (quando aplicável). |
| `ChegadaPrevista` | `datetime64[ns]` | Horário programado de pouso | Atributo base para cálculo do tempo de voo previsto. |
| `ChegadaReal` | `datetime64[ns]` | Horário efetivo de pouso | **Variável estritamente pós-fato.** Utilizada apenas no cálculo do alvo; **excluída de $X$**. |
| `SituacaoVoo` | `string / category` | `REALIZADO` ou `CANCELADO` | Filtro de escopo: reter `REALIZADO` para predição de atraso. |
| `CodigoJustificativa` | `string / category` | Motivo operacional do atraso | **Vazamento de dados direto.** Exclusão mandatória do vetor $X$. |

### 2.2 Política Rígida Anti-Data Leakage (Blindagem de Vazamento)
Conforme ensinado nos slides de *Parametrização* e no *Guia Metodológico*:
1. **Regra de Ouro:** Transformadores como `StandardScaler` e `OneHotEncoder` devem ser instanciados e ajustados (`fit`) **apenas** sobre a partição de Treinamento ($D_{\text{treino}}$). As partições de Validação ($D_{\text{val}}$) e Teste ($D_{\text{teste}}$) recebem apenas o método `transform()`.
2. **Eliminação de Variáveis Concorrentes:** Qualquer coluna alimentada durante a execução ou conclusão do voo (`ChegadaReal`, `SituacaoVoo` pós-voo, `CodigoJustificativa`) é removida da matriz preditora $X$.
3. **Divisão Cronológica Obrigatória:** É proibido qualquer embaralhamento pseudoaleatório (`shuffle=True`). Os voos são ordenados por `PartidaPrevista` em ordem ascendente.

### 2.3 Engenharia de Features Derivadas
A partir das colunas brutas autorizadas, o pipeline calcula os seguintes atributos:

$$\text{TempoVooPrevistoMin} = \frac{\text{ChegadaPrevista} - \text{PartidaPrevista}}{60 \text{ segundos}}$$

$$\text{HoraPartida} = \text{PartidaPrevista.hour} + \frac{\text{PartidaPrevista.minute}}{60}$$

$$\text{Hora}_{\sin} = \sin\left(\frac{2\pi \cdot \text{HoraPartida}}{24}\right), \quad \text{Hora}_{\cos} = \cos\left(\frac{2\pi \cdot \text{HoraPartida}}{24}\right)$$

$$\text{DiaSemana}_{\sin} = \sin\left(\frac{2\pi \cdot \text{PartidaPrevista.dayofweek}}{7}\right), \quad \text{DiaSemana}_{\cos} = \cos\left(\frac{2\pi \cdot \text{PartidaPrevista.dayofweek}}{7}\right)$$

$$\text{Mes}_{\sin} = \sin\left(\frac{2\pi \cdot (\text{PartidaPrevista.month} - 1)}{12}\right), \quad \text{Mes}_{\cos} = \cos\left(\frac{2\pi \cdot (\text{PartidaPrevista.month} - 1)}{12}\right)$$

$$\text{Rota} = \text{ICAOAerodromoOrigem} + \text{'_'} + \text{ICAOAerodromoDestino}$$

### 2.4 Definição Formal da Variável Alvo ($y$)
- **Problema Principal (Classificação Binária):**
  $$\text{AtrasoChegadaMin} = \frac{\text{ChegadaReal} - \text{ChegadaPrevista}}{60 \text{ segundos}}$$
  $$y = \begin{cases} 1, & \text{se } \text{AtrasoChegadaMin} > 15 \\ 0, & \text{se } \text{AtrasoChegadaMin} \le 15 \end{cases}$$
- **Problema Secundário (Regressão Contínua, se aplicável):**
  $$y_{\text{reg}} = \max(0, \text{AtrasoChegadaMin}) \in \mathbb{R}^+$$

### 2.5 Particionamento Experimental dos Dados
- **Treinamento ($D_{\text{treino}}$):** Primeiros $70\%$ cronológicos. Utilizado para ajuste de pesos e parâmetros de escala.
- **Validação ($D_{\text{val}}$):** $15\%$ intermediários cronológicos. Utilizado para escolha de hiperparâmetros no Optuna, limiar de corte e Early Stopping.
- **Teste ($D_{\text{teste}}$):** $15\%$ finais mais recentes cronológicos. Utilizado para avaliação final cega da generalização.

---

## 3. SPEC-02: Especificação de Arquitetura de Software (`src/`)

O projeto deve ser estruturado em módulos Python puros, testáveis e desacoplados, consumíveis tanto pelos scripts CLI quanto pelos notebooks:

```text
src/
├── __init__.py
├── data_pipeline.py   # Ingestão, higienização, splits temporais e ColumnTransformer
├── models.py          # Fábrica de modelos Keras (MLP, RNN, Autoencoder)
├── tuning.py          # Integração do Optuna, Keras Callback e pruning
└── evaluation.py      # Cálculo de métricas desbalanceadas, gráficos e custos
```

### 3.1 Módulo `src/data_pipeline.py`
```python
# Assinaturas Obrigatórias
def load_vra_raw(filepath: str) -> pd.DataFrame:
    """Carrega dados brutos do VRA tratando delimitadores e encodings (latin1/utf-8)."""
    ...

def clean_and_feature_engineer(df: pd.DataFrame, delay_threshold_min: int = 15) -> pd.DataFrame:
    """Filtra situações válidas, remove vazamentos de dados, extrai variáveis cíclicas
    e computa a variável alvo binária."""
    ...

def split_chronological(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Realiza partição temporal ordenada estrita (Train, Val, Test)."""
    ...

def build_and_fit_preprocessor(train_df: pd.DataFrame, top_k_airports: int = 20):
    """Instancia e ajusta (fit) ColumnTransformer EXCLUSIVAMENTE em train_df.
    Retorna o transformer ajustado e os metadados de colunas."""
    ...

def transform_splits(preprocessor, train_df, val_df, test_df):
    """Aplica transform() uniforme nas partições sem vazar estatísticas."""
    ...
```

### 3.2 Módulo `src/models.py`
Todas as camadas, inicializações e funções de ativação seguem estritamente as aulas do Prof. Sérgio Canuto:
- **Ativações:** $\text{ReLU}$ para camadas intermediárias; $\text{Sigmoid}$ para classificação binária; $\text{Linear}$ para regressão.
- **Inicializadores:** He Normal para neurônios $\text{ReLU}$; Glorot Uniform para $\text{Sigmoid}$.
- **Regularizadores:** Dropout (`tf.keras.layers.Dropout`) e Penalização $\text{L}_2$ (`tf.keras.regularizers.l2`).
- **Normalização Interna:** `tf.keras.layers.BatchNormalization`.

```python
# Assinaturas Obrigatórias
def create_mlp_model(input_dim: int, 
                     n_layers: int = 2, 
                     units: int = 128, 
                     dropout_rate: float = 0.2, 
                     l2_reg: float = 1e-4, 
                     learning_rate: float = 1e-3, 
                     use_batch_norm: bool = True) -> tf.keras.Model:
    """Constrói MLP densa compilada com Binary Cross-Entropy e otimizador Adam."""
    ...

def create_recurrent_model(input_shape: tuple, 
                           cell_type: str = 'LSTM',  # 'SimpleRNN', 'GRU', 'LSTM'
                           units: int = 64, 
                           n_layers: int = 1, 
                           dropout_rate: float = 0.2, 
                           learning_rate: float = 1e-3) -> tf.keras.Model:
    """Constrói rede recorrente comparativa (SimpleRNN vs GRU vs LSTM) para séries VRA."""
    ...

def create_autoencoder_model(input_dim: int, 
                             latent_dim: int = 2, 
                             hidden_dims: list = [64, 32], 
                             learning_rate: float = 1e-3):
    """Constrói Autoencoder subcompleto retornando (autoencoder_completo, encoder_extrator)."""
    ...
```

### 3.3 Módulo `src/tuning.py`
Encapsula o framework **Optuna** com suporte a parada antecipada de tentativas infrutíferas (*pruning*):

```python
# Assinaturas Obrigatórias
class KerasPruningCallback(tf.keras.callbacks.Callback):
    """Comunica a métrica de validação por época ao Optuna e dispara trial.should_prune()."""
    ...

def objective_mlp(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """Função objetivo para busca de hiperparâmetros da MLP densa.
    Otimiza val_loss ou val_macro_f1 calculados estritamente no conjunto de validação."""
    ...

def run_study(objective_func, n_trials: int = 30, study_name: str = "vra_dl_study") -> optuna.Study:
    """Cria e executa estudo Optuna utilizando TPESampler e MedianPruner."""
    ...
```

### 3.4 Módulo `src/evaluation.py`
Gera todas as tabelas e gráficos exigidos pelo Edital (Seção 2.5 e Seção 6):

```python
# Assinaturas Obrigatórias
def compute_comprehensive_metrics(y_true, y_pred_prob, threshold: float = 0.5) -> dict:
    """Calcula Macro-F1, Precision, Recall, Accuracy, AUC-ROC e Log-Loss."""
    ...

def plot_learning_curves(history: tf.keras.callbacks.History, save_path: str = None):
    """Gera gráficos de Loss (Treino vs Validação) e Métricas para diagnóstico de Overfitting."""
    ...

def plot_confusion_matrix(y_true, y_pred, labels=['Pontual', 'Atrasado'], save_path: str = None):
    """Gera Matriz de Confusão com contagens absolutas e percentuais normalizados."""
    ...

def benchmark_inference_efficiency(model: tf.keras.Model, X_sample, n_iterations: int = 50) -> dict:
    """Mede número de parâmetros treináveis, tempo médio de inferência para 1.000 amostras
    e pegada computacional."""
    ...
```

---

## 4. SPEC-03: Especificação da Otimização Bayesiana com Optuna

### 4.1 Justificativa Teórica do Espaço de Busca
O Edital explicita na Seção 2.4 que o grupo não pode apenas exibir os números finais; deve justificar a escolha de cada faixa de busca:

| Hiperparâmetro | Distribuição Optuna | Intervalo Definido | Justificativa Teórica (Aulas Prof. Canuto) |
| :--- | :--- | :--- | :--- |
| `learning_rate` ($\eta$) | `trial.suggest_float(..., log=True)` | $[10^{-4}, 10^{-1}]$ | Escala logarítmica é necessária porque variações de ordem de magnitude impactam drasticamente a estabilidade dos gradientes. |
| `batch_size` ($B$) | `trial.suggest_categorical(...)` | $\{32, 64, 128, 256\}$ | Lotes menores geram ruído de gradiente benéfico para escapar de mínimos locais rasos; lotes maiores exploram paralelismo matricial. |
| `n_layers` ($L$) | `trial.suggest_int(...)` | $[1, 4]$ | Avalia a necessidade de profundidade representacional vs. risco de sobreajuste e saturação de gradientes. |
| `units` | `trial.suggest_categorical(...)` | $\{32, 64, 128, 256\}$ | Define a largura e dimensionalidade do espaço de hipóteses da camada oculta. |
| `dropout_rate` ($p$) | `trial.suggest_float(...)` | $[0.0, 0.5]$ | Regularização estocástica por desativação de conexões para evitar coadaptação complexa entre neurônios. |
| `l2_reg` ($\lambda$) | `trial.suggest_float(..., log=True)` | $[10^{-6}, 10^{-2}]$ | Penalidade de decaimento de pesos (Weight Decay) para comprimir coeficientes excessivamente grandes. |

### 4.2 Protocolo de Poda e Critério de Parada
- **Sampler:** `TPESampler(seed=42)` (Tree-structured Parzen Estimator).
- **Pruner:** `MedianPruner(n_startup_trials=5, n_warmup_steps=5)` para descartar *trials* cujo desempenho na época intermediária seja inferior à mediana histórica.
- **Early Stopping nos Modelos:** `EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)`.

---

## 5. SPEC-04: Protocolo Experimental e Métricas de Avaliação

### 5.1 Tratamento do Desbalanceamento
A base de voos possui preponderância de operações pontuais. A acurácia isolada é insuficiente.

1. **Métrica Primária:** $\text{Macro F1-Score}$
   $$\text{F1}_{\text{Macro}} = \frac{\text{F1}_{\text{classe 0}} + \text{F1}_{\text{classe 1}}}{2}$$
2. **Métricas de Discriminação:** Área sob a curva ROC ($\text{AUC-ROC}$) e Curva Precision-Recall ($\text{PR-AUC}$).
3. **Ponderação de Classes (Class Weights):**
   $$w_0 = \frac{N}{2 \cdot N_0}, \quad w_1 = \frac{N}{2 \cdot N_1}$$
   Passado ao método `model.fit(..., class_weight={0: w0, 1: w1})` para penalizar proporcionalmente os erros na classe minoritária (voos com atraso crítico).

### 5.2 Diagnóstico Visual de Curvas de Treinamento
Os gráficos de curvas de aprendizado devem classificar inequivocamente:
- **Subajuste (Underfitting):** Perda de treinamento e de validação estabilizadas em patamares elevados com baixo aprendizado.
- **Sobreajuste (Overfitting):** Perda de treinamento decrescente contínua, enquanto a perda de validação atinge ponto de inflexão e passa a divergir ascendentemente.
- **Ajuste Equilibrado (Good Fit):** Convergência suave e próxima entre curvas de treino e validação graças à ação de Dropout e Early Stopping.

### 5.3 Análise de Custo-Efetividade Computacional
O modelo final deve reportar a tabela comparativa:
- $\text{Total de Parâmetros}$ e $\text{Parâmetros Treináveis}$.
- $\text{Tempo de Treinamento por Época (s)}$.
- $\text{Latência de Inferência (ms por 1.000 amostras)}$.
- $\text{Desempenho no Teste (Macro-F1)}$.

---

## 6. SPEC-05: Especificação da Frente Principal (MLP) e Frente Comparativa (Redes Recorrentes)

Conforme deliberação do projeto, o trabalho implementa como foco central a **Frente 3.1 (Multilayer Perceptrons)** e realiza um estudo comparativo direto com a **Frente 3.2 (Redes Recorrentes - LSTM/GRU)**:

### 6.1 Frente Principal: Multilayer Perceptrons (Frente 3.1)
- Modelagem preditiva dos registros individuais com o pipeline tabular e variáveis cíclicas.
- Construção e teste de $3$ variações estruturadas:
  1. *MLP Simples (Baseline):* 1 camada oculta ($64$ neurônios), sem dropout.
  2. *MLP Profunda com Regularização:* $3$ camadas ocultas ($256 \to 128 \to 64$), Dropout $0.3$, Batch Normalization e $\text{L}_2$.
  3. *MLP Otimizada Optuna:* Configuração selecionada pelo algoritmo bayesiano como a de menor perda na validação.
- Estudo completo com Optuna variando learning rate (log), profundidade, neurônios, dropout e batch size.

### 6.2 Frente Comparativa: Redes Recorrentes (Frente 3.2)
- Modelagem de séries temporais multivariadas com janelas históricas de voos de uma mesma rota ou aeroporto ($W \in \{5, 10\}$ voos anteriores).
- Comparação direta sob o mesmo protocolo experimental: `SimpleRNN` vs. `GRU` vs. `LSTM`.
- Análise de trade-off: comparar se a representação sequencial oferece ganho de Macro-F1 que justifique o aumento de parâmetros, tempo de treinamento e latência de inferência em relação à MLP principal.

---

## 7. SPEC-06: Critérios de Aceitação e Checklist dos Entregáveis

- [ ] **Código Fonte:** Repositório no GitHub contendo `src/`, `notebooks/`, `requirements.txt` e `README.md` detalhado com instruções de reprodução.
- [ ] **Rastreabilidade de Hiperparâmetros:** Registro em arquivo persistente do estudo Optuna, tabela de parâmetros explorados e justificativa formal de cada intervalo.
- [ ] **Curvas de Aprendizado:** Gráficos de perda e métricas salvos e analisados criticamente.
- [ ] **Relatório Técnico Acadêmico:** Redigido nas 6 seções obrigatórias especificadas na Seção 5 do Edital.
- [ ] **Slides de Apresentação:** Estruturados para defesa rigorosa dentro do limite estrito de **10 minutos** com a distribuição de tempo especificada no plano.
