# Trabalho Prático de Deep Learning: Previsão de Atrasos no Voo Regular Ativo (VRA / ANAC)
## Modelagem, Hiperparametrização e Avaliação Experimental

Repositório acadêmico desenvolvido para a disciplina de **Deep Learning (2026/2)** ministrada pelo **Prof. Dr. Sérgio Canuto** (IFG).

---

## 1. Visão Geral do Projeto

Este projeto desenvolve uma solução de Deep Learning aplicada aos microdados abertos de **Voo Regular Ativo (VRA)** da Agência Nacional de Aviação Civil (ANAC). 

### Frentes Tecnológicas Avaliadas:
- **Frente Principal (3.1): Multilayer Perceptrons (MLP)** — Investigação de profundidade, largura, regularização (Dropout, $\text{L}_2$, Batch Normalization) e otimização bayesiana com Optuna.
- **Frente Comparativa (3.2): Redes Recorrentes (RNN / LSTM / GRU)** — Avaliação comparativa de modelagem temporal sequencial em janelas históricas de voos vs. a representação tabular densa da MLP.

---

## 2. Estrutura do Repositório

```text
deep_learning_trabalho_final/
├── data/
│   ├── raw/             # Microdados brutos do VRA (CSV/Dados Abertos)
│   └── processed/       # Partições pré-processadas e transformadores salvos
├── notebooks/
│   ├── 01_eda_vra.ipynb               # Análise exploratória, nulos e desbalanceamento
│   ├── 02_preprocessing_splits.ipynb  # Divisão cronológica e pipeline anti-leakage
│   ├── 03_optuna_experiments.ipynb    # Otimização Bayesiana da MLP com Optuna
│   └── 04_final_evaluation.ipynb      # Teste cego, curvas de loss e comparativo RNN
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py # Ingestão, features cíclicas e ColumnTransformer
│   ├── models.py        # Fábrica de modelos Keras (MLP, LSTM, GRU, SimpleRNN)
│   ├── tuning.py        # Objetivos Optuna, Keras Callback e MedianPruner
│   └── evaluation.py    # Macro-F1, matriz de confusão e custo computacional
├── reports/
│   ├── relatorio_tecnico.pdf          # Relatório acadêmico completo (6 seções)
│   └── slides_apresentacao.pdf        # Apresentação de defesa de 10 minutos
├── specs/                             # Especificações técnicas e plano versionado
│   ├── plano_implementacao_versionado.md
│   ├── specs_detalhadas.md
│   ├── spec_01_dados_pipeline.md
│   ├── spec_02_arquiteturas_modelagem.md
│   ├── spec_03_optuna_hiperparametrizacao.md
│   ├── spec_04_avaliacao_experimentos.md
│   └── spec_05_relatorio_apresentacao.md
├── documentos/          # Base de slides e editais da disciplina (RAG fechado)
├── requirements.txt     # Dependências exatas e reprodutíveis
├── .gitignore
└── README.md
```

---

## 3. Instruções de Instalação e Execução

### 3.1 Clonagem e Criação do Ambiente Virtual
```bash
# 1. Clonar repositório
git clone <url-do-repositorio>
cd deep_learning_trabalho_final

# 2. Criar e ativar ambiente virtual
python -m venv .venv
# No Linux/macOS:
source .venv/bin/activate
# No Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Instalar dependências
pip install -r requirements.txt
```

### 3.2 Execução dos Notebooks Experimentais
Para reproduzir todos os experimentos, execute os notebooks na ordem cronológica:
1. `notebooks/01_eda_vra.ipynb`: Carrega a base bruta e diagnostica distribuições e desbalanceamento.
2. `notebooks/02_preprocessing_splits.ipynb`: Aplica a regra anti-leakage com split cronológico (70/15/15).
3. `notebooks/03_optuna_experiments.ipynb`: Dispara a busca bayesiana da MLP com Optuna e poda inteligente.
4. `notebooks/04_final_evaluation.ipynb`: Avalia os modelos vencedores no teste cego e executa o comparativo temporal com LSTM/GRU.

---

## 4. Conformidade Rigorosa com o Edital

- **Anti-Data Leakage:** Estimadores (`StandardScaler`, `OneHotEncoder`) ajustados unicamente na partição de Treinamento.
- **Divisão Cronológica:** 70% Treino, 15% Validação e 15% Teste sem embaralhamento (`shuffle=False`).
- **Métrica Desbalanceada:** Adoção de **Macro F1-Score**, AUC-ROC e Precision-Recall.
- **Análise Custo-Efetividade:** Relacionamento entre parâmetros treináveis, tempo por época e latência de inferência (ms por 1.000 amostras).
