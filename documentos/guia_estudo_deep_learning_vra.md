# Guia de Estudo e Base de Conhecimento Científica: Deep Learning no VRA/ANAC
> **Projeto de Final de Curso — Disciplina de Deep Learning (2026/2)**  
> **Docente:** Prof. Dr. Sérgio Canuto | **Instituto Federal de Goiás (IFG)**  
> **Tema:** Modelagem Preditiva de Atrasos Operacionais no Voo Regular Ativo (VRA/ANAC)

Este documento foi elaborado para transformar a teoria de Deep Learning em conceitos claros, intuitivos e diretamente mapeados para o código-fonte do projeto. Use-o para estudar, se preparar para perguntas do professor e explicar cada decisão técnica com rigor e clareza para seus colegas.

---

## Sumário dos Tópicos de Estudo

1. [Tópico 1: O Problema Real e a Formulação Matemática](#tópico-1-o-problema-real-e-a-formulação-matemática)
2. [Tópico 2: A Regra de Ouro do Pré-processamento e Prevenção de Data Leakage](#tópico-2-a-regra-de-ouro-do-pré-processamento-e-prevenção-de-data-leakage)
3. [Tópico 3: Engenharia de Atributos e Codificação Cíclica (Seno e Cosseno)](#tópico-3-engenharia-de-atributos-e-codificação-cíclica-seno-e-cosseno)
4. [Tópico 4: Divisão Cronológica e Tratamento do Desbalanceamento (Class Weights)](#tópico-4-divisão-cronológica-e-tratamento-do-desbalanceamento-class-weights)
5. [Tópico 5: Arquitetura Multilayer Perceptron (MLP) — Frente Principal 3.1](#tópico-5-arquitetura-multilayer-perceptron-mlp--frente-principal-31)
6. [Tópico 6: Redes Neurais Recorrentes (LSTM) — Estudo Comparativo 3.2](#tópico-6-redes-neurais-recorrentes-lstm--estudo-comparativo-32)
7. [Tópico 7: Otimização Bayesiana de Hiperparâmetros com Optuna](#tópico-7-otimização-bayesiana-de-hiperparâmetros-com-optuna)
8. [Tópico 8: Métricas de Avaliação e Análise de Custo-Efetividade Computacional](#tópico-8-métricas-de-avaliação-e-análise-de-custo-efetividade-computacional)
9. [Tópico 9: Como Executar Novos Experimentos e Buscar Resultados Superiores](#tópico-9-como-executar-novos-experimentos-e-buscar-resultados-superiores)

---

## Tópico 1: O Problema Real e a Formulação Matemática

### 1. Contexto e Intuição
No setor aéreo, um atraso de voo acarreta efeito cascata (conexões perdidas, tripulação excedendo limites de horas, aeronaves paradas na pista gerando multas). Prever com antecedência se um voo vai atrasar criticamente permite que os controladores e companhias aéreas reestruturem rotas antes mesmo da decolagem.

- **Unidade de Observação:** Cada linha representa um **voo regular comercial individual**.
- **Limiar Crítico Regulatório:** A ANAC e a IATA definem que atrasos de até 15 minutos são tolerância operacional normal (embarque de bagagens, fila de táxi). Acima de 15 minutos é considerado **atraso crítico**.

### 2. Formulação Matemática
$$\text{AtrasoChegadaMin} = \frac{\text{DataHoraReal}_{\text{chegada}} - \text{DataHoraPrevista}_{\text{chegada}}}{60 \text{ segundos}}$$

A variável alvo binária $y \in \{0, 1\}$ é formulada como:
$$y = \begin{cases} 1, & \text{se } \text{AtrasoChegadaMin} > 15 \text{ minutos (Atraso Crítico)} \\ 0, & \text{se } \text{AtrasoChegadaMin} \le 15 \text{ minutos (Pontual/Adiantado)} \end{cases}$$

### 3. Onde está no Código?
- Arquivo: [`src/data_pipeline.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/data_pipeline.py#L54-L60)
- Linhas: `54-60` na função `clean_and_prepare_vra()`:
  ```python
  df_clean['AtrasoChegadaMin'] = (df_clean['ChegadaReal'] - df_clean['ChegadaPrevista']).dt.total_seconds() / 60.0
  df_clean = df_clean[(df_clean['AtrasoChegadaMin'] >= -120) & (df_clean['AtrasoChegadaMin'] <= 1440)].copy()
  df_clean['target_bin'] = (df_clean['AtrasoChegadaMin'] > delay_threshold_min).astype(int)
  ```

### 4. Como Explicar aos Colegas
> *"Nós não tentamos adivinhar os minutos exatos com regressão porque, na operação real, saber se vai atrasar 18 ou 21 minutos importa menos do que a decisão binária de acionar um plano de contingência para atrasos superiores aos 15 minutos tolerados pela ANAC."*

---

## Tópico 2: A Regra de Ouro do Pré-processamento e Prevenção de Data Leakage

### 1. O Conceito de Data Leakage (Vazamento de Dados)
Vazamento de dados ocorre quando informações que só existiriam **depois** do evento ou informações do conjunto de teste/validação são inadvertidamente injetadas durante o treinamento. O modelo parece atingir 99% de acurácia em sala de aula, mas falha catastróficamente em produção.

### 2. Duas Fontes Graves Evitadas no Projeto
1. **Variáveis Concorrentes/Futuras:** Colunas como `ChegadaReal`, `PartidaReal` e `CodigoJustificativa`. Se a rede receber `PartidaReal`, ela simplesmente subtrai da prevista e descobre o atraso com trapaça. Portanto, elas foram **eliminadas** da matriz preditora $X$.
2. **Ajuste Estatístico (Fit Leakage):** Se você calcular a média e desvio padrão (`StandardScaler`) em toda a base antes de dividir em treino e teste, o treino estará "espiando" a distribuição de dados do futuro.

### 3. A Regra de Ouro Implementada
$$\mu_{\text{treino}}, \sigma_{\text{treino}} \leftarrow \text{Fit exclusivamente em } D_{\text{treino}}$$
$$X_{\text{val}} = \frac{X_{\text{val}} - \mu_{\text{treino}}}{\sigma_{\text{treino}}}, \quad X_{\text{teste}} = \frac{X_{\text{teste}} - \mu_{\text{treino}}}{\sigma_{\text{treino}}}$$

### 4. Onde está no Código?
- Arquivo: [`src/data_pipeline.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/data_pipeline.py#L117-L154)
- Linhas: `121-132` (função `build_and_fit_preprocessor`) e `134-154` (função `transform_splits`):
  ```python
  # O preprocessor faz fit APENAS no train_df:
  preprocessor.fit(train_df[num_cols + cat_cols])
  
  # As outras partições apenas recebem transform():
  X_train = preprocessor.transform(train_df[feature_cols])
  X_val = preprocessor.transform(val_df[feature_cols])
  X_test = preprocessor.transform(test_df[feature_cols])
  ```
- Material de Referência: Slide *"Deep Learning - parametrizacao 2026_2 (1).pdf"* (Seção de Particionamento e Normalização).

---

## Tópico 3: Engenharia de Atributos e Codificação Cíclica (Seno e Cosseno)

### 1. O Problema da Descontinuidade Numérica
Se representarmos a hora do dia como um número linear de $0$ a $23$:
- O voo programado para as **23:55** tem valor $\approx 23.9$.
- O voo programado para as **00:05** tem valor $\approx 0.08$.
- A distância aritmética entre eles é $|23.9 - 0.08| = 23.82$ horas! No entanto, na realidade física eles estão separados por apenas **10 minutos**! Uma rede neural linear interpretaria isso como extremos opostos.

### 2. A Solução: Projeção Circular Trigonométrica
Transformamos o tempo linear em um ângulo de um círculo unitário ($2\pi$ radianos):
$$\text{Hora}_{\sin} = \sin\left(\frac{2\pi \cdot \text{Hora}}{24}\right), \quad \text{Hora}_{\cos} = \cos\left(\frac{2\pi \cdot \text{Hora}}{24}\right)$$
$$\text{DiaSemana}_{\sin} = \sin\left(\frac{2\pi \cdot \text{Dia}}{7}\right), \quad \text{DiaSemana}_{\cos} = \cos\left(\frac{2\pi \cdot \text{Dia}}{7}\right)$$

Com o par $(\sin, \cos)$, o ponto $(23:55)$ e o ponto $(00:05)$ ficam geometricamente adjacentes na circunferência, preservando a continuidade temporal exata para a rede neural.

### 3. Redução de Cardinalidade Categórica
Aeroportos e companhias aéreas possuem dezenas de siglas que aparecem apenas 1 ou 2 vezes ao ano. Usar One-Hot Encoding puro criaria centenas de colunas esparsas e levaria a overfitting imediato.
- **Técnica:** Mantivemos apenas as top 10 companhias e top 20 aeroportos mais frequentes; todas as outras ocorrências raras foram agrupadas no rótulo `'OUTROS'`.

### 4. Onde está no Código?
- Arquivo: [`src/data_pipeline.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/data_pipeline.py#L65-L89)
- Linhas: `65-79` (cálculo de seno/cosseno) e `81-89` (agrupamento top-k).

---

## Tópico 4: Divisão Cronológica e Tratamento do Desbalanceamento (Class Weights)

### 1. Por que NÃO usar K-Fold aleatório ou `train_test_split(shuffle=True)`?
Em séries temporais e dados operacionais de voo, o passado prevê o futuro. Se você embaralhar as linhas aleatoriamente, a rede usará atrasos ocorridos em dezembro para prever voos de janeiro do mesmo ano. Isso é previsão com dados do futuro.
- **Divisão temporal estrita:**
  * **Treinamento ($70\%$):** Voos mais antigos cronologicamente ($D_{\text{treino}}$).
  * **Validação ($15\%$):** Período intermediário (usado pelo Optuna e Early Stopping).
  * **Teste ($15\%$):** Período mais recente (teste cego, intocado durante todo o treino).

### 2. Desbalanceamento da Base e Class Weights
A pontualidade da aviação brasileira é de aproximadamente $78.8\%$ e os voos atrasados representam $21.2\%$.
- Se treinarmos uma rede convencional com perda ordinária, a rede aprenderá o caminho de menor esforço: prever **sempre $0$ (pontual)** e obter $78.8\%$ de acurácia, mas acertando zero atrasos críticos.
- **Solução Ponderada na Função de Perda (Binary Cross-Entropy Ponderada):**
  $$w_0 = \frac{N}{2 \cdot N_0} = 0.637, \quad w_1 = \frac{N}{2 \cdot N_1} = 2.325$$
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ w_1 y_i \log(\hat{y}_i) + w_0 (1 - y_i) \log(1 - \hat{y}_i) \right]$$
  Errar um voo atrasado ($y=1$) custa para a rede **3.65 vezes mais caro** do que errar um voo pontual, forçando o gradiente a aprender os padrões dos atrasos.

### 3. Onde está no Código?
- Arquivo: [`main.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/main.py#L51-L77)
- Linhas: `51-56` (divisão cronológica) e `70-76` (cálculo de pesos de classe).

---

## Tópico 5: Arquitetura Multilayer Perceptron (MLP) — Frente Principal 3.1

### 1. Teorema da Aproximação Universal e Fundamentos
O Teorema da Aproximação Universal (Hornik, Stinchcombe & White, 1989) demonstra que uma rede neural feedforward com uma única camada oculta e ativações não-lineares pode aproximar qualquer função contínua em espaços compactos. No entanto, redes **profundas** conseguem representar funções complexas com exponencialmente menos parâmetros do que redes rasas excessivamente largas.

### 2. Elementos Críticos da Camada Oculta
1. **Ativação ReLU ($\text{ReLU}(z) = \max(0, z)$):** Elimina a saturação de gradiente típica de sigmoide/tangente hiperbólica em camadas profundas, permitindo convergência rápida via SGD/Adam.
2. **Inicialização He Normal:** Os pesos são inicializados com variância $\sigma^2 = \frac{2}{n_{\text{in}}}$, garantindo que a variância das ativações permaneça estável ao longo de múltiplas camadas ReLU.
3. **Batch Normalization (Ioffe & Szegedy, 2015):** Normaliza a saída de cada camada oculta para média zero e variância unitária antes da ativação, reduzindo o *Internal Covariate Shift* e acelerando drasticamente o aprendizado.
4. **Dropout (Srivastava et al., 2014):** Durante o treino, desativa aleatoriamente uma proporção $p$ dos neurônios a cada passagem, impedindo que neurônios coadaptem seus pesos em conjunto.
5. **Decaimento de Pesos L2 ($\frac{\lambda}{2} \sum \|W\|^2$):** Penaliza coeficientes com magnitudes desproporcionais, gerando superfícies de decisão suaves.

### 3. As Três Variantes Experimentais no Projeto
- **V1 (Rasa Baseline):** 1 camada de 64 neurônios, sem Dropout, sem L2, sem BatchNorm. (Macro-F1: 0.4348, 2.433 parâmetros).
- **V2 (Profunda Regularizada):** 3 camadas ($256 \to 128 \to 64$), BatchNorm em todas, Dropout(0.3) e L2 ($10^{-4}$). (Macro-F1: **0.4936** — Campeã no equilíbrio das classes).
- **V3 (Otimizada Optuna):** Topologia selecionada por busca bayesiana ($32 \to 64 \to 128$), Dropout(0.216), L2 ($1.46 \times 10^{-5}$). (AUC-ROC: **0.5472**, 12.193 parâmetros).

### 4. Onde está no Código?
- Arquivo: [`src/models.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/models.py#L10-L60)
- Linhas: `10-60` na função `build_mlp()`:
  ```python
  for idx, units in enumerate(hidden_units):
      reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
      x = layers.Dense(units, kernel_initializer='he_normal', kernel_regularizer=reg)(x)
      if use_batch_norm:
          x = layers.BatchNormalization()(x)
      x = layers.Activation('relu')(x)
      if dropout_rate > 0:
          x = layers.Dropout(dropout_rate)(x)
  ```
- Material de Referência: Slides *"Deep Learning - Neurônios Artificiais 2026_2"* e *"Deep Learning - Backpropagation 2026_2"*.

---

## Tópico 6: Redes Neurais Recorrentes (LSTM) — Estudo Comparativo 3.2

### 1. Por que testar Redes Recorrentes?
Voos operam ao longo do tempo. Uma hipótese intuitiva é: se os últimos 5 voos de uma companhia ou aeroporto atrasaram, o sexto voo provavelmente atrasará devido a congestionamento de pátio ou fila de aeronaves. Para testar essa hipótese, criamos tensores tridimensionais com janela deslizante de tamanho $W = 5$:
$$\text{Formato do Tensor 3D:} \quad (\text{Número de Voos}, \text{Janela Temporal}=5, \text{Atributos}=36)$$

### 2. Mecanismo da Célula LSTM (Long Short-Term Memory)
Redes Recorrentes Simples sofrem com o problema do desaparecimento do gradiente (*Vanishing Gradient*): em retropropagação através do tempo (BPTT), multiplicações repetidas de matrizes fazem o gradiente tender a zero. A LSTM soluciona isso criando uma "autoestrada aditiva" de fluxo de gradiente através do **Estado de Célula** ($c_t$):
- **Porta de Esquecimento ($f_t$):** Decide qual fração do histórico anterior deve ser descartada:
  $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
- **Porta de Entrada ($i_t, \tilde{c}_t$):** Decide quais novas informações do voo atual são relevantes:
  $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i), \quad \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
- **Atualização do Estado de Célula ($c_t$):**
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
- **Porta de Saída e Estado Oculto ($o_t, h_t$):**
  $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o), \quad h_t = o_t \odot \tanh(c_t)$$

### 3. Veredito Científico: Por que a MLP superou a LSTM?
- **Desempenho:** A LSTM atingiu Macro-F1 de **0.4443**, inferior aos **0.4936** da MLP V2.
- **Custo Computacional:** A latência da LSTM foi **316.30 ms por 1k amostras** — $52\%$ mais lenta que a MLP V2 (207.89 ms).
- **Justificativa Operacional:** No sistema aeroportuário nacional, voos não ocorrem em uma linha temporal unívoca e serializada. Centenas de voos partem simultaneamente de dezenas de pistas diferentes (Brasília, Guarulhos, Confins). Forçar essa malha paralela em uma cadeia unidimensional linear gera ruído temporal espúrio. Logo, a **modelagem tabular densa (MLP) é superior em custo-efetividade**.

### 4. Onde está no Código?
- Criação dos Tensores 3D: [`src/data_pipeline.py:L156-L171`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/data_pipeline.py#L156-L171) (`create_recurrent_sequences`).
- Arquitetura LSTM: [`src/models.py:L61-L96`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/models.py#L61-L96) (`build_recurrent_classifier`).
- Material de Referência: Slide *"Deep Learning - Redes recorrentes 2026_2.pdf"*.

---

## Tópico 7: Otimização Bayesiana de Hiperparâmetros com Optuna

### 1. Grid Search vs. Random Search vs. Otimização Bayesiana (TPE)
- **Grid Search:** Testa todas as combinações em grade cartesiana. É extremamente ineficiente e desperdiça tempo avaliando hiperparâmetros irrelevantes.
- **Random Search:** Amostra pontos aleatórios. Melhora a exploração, mas é completamente "cego": não aprende com os erros passados.
- **Otimização Bayesiana (TPE - Tree-structured Parzen Estimator):** Constrói um modelo probabilístico $p(\theta | y)$ a cada iteração. Se baixas taxas de aprendizado e tamanhos de lote moderados produziram perdas menores nas tentativas anteriores, o algoritmo concentra as próximas amostragens nessa região promissora do espaço de busca.

### 2. Poda Antecipada (MedianPruner) e Early Stopping
Treinar redes neurais até a última época quando a curva de validação já está divergindo é desperdício de GPU/CPU.
- **OptunaPruningCallback:** A cada época, a perda de validação é reportada ao Optuna. Se a curva do modelo atual estiver pior do que a mediana das tentativas anteriores naquele mesmo passo, o Optuna interrompe o treino imediatamente com `TrialPruned`.
- **EarlyStopping:** Interrompe se o `val_loss` não melhorar por 5 épocas consecutivas e restaura os melhores pesos.

### 3. O Espaço de Busca e Justificativas Teóricas
| Hiperparâmetro | Intervalo / Opções | Escala | Justificativa Teórica |
| :--- | :--- | :--- | :--- |
| **Taxa de Aprendizado ($\eta$)** | $[10^{-4}, 10^{-1}]$ | Logarítmica | Passo de atualização do gradiente. Escala log garante exploração uniforme em ordens de grandeza ($0.0001, 0.001, 0.01$). |
| **Tamanho do Lote ($B$)** | $\{32, 64, 128, 256\}$ | Categórica | Lotes menores geram ruído no gradiente que atua como regularizador; lotes maiores aceleram o throughput matricial. |
| **Profundidade ($L$)** | $[1, 4]$ camadas | Inteiro Discreto | Avalia a transição entre modelos rasos e redes profundas. |
| **Neurônios por Camada** | $\{32, 64, 128, 256\}$ | Categórica | Capacidade representacional de cada nível hierárquico. |
| **Taxa de Dropout ($p$)** | $[0.0, 0.5]$ | Uniforme Contínua | Regularização por esparsidade estocástica. |
| **Decaimento L2 ($\lambda$)** | $[10^{-6}, 10^{-2}]$ | Logarítmica | Regularização de peso que restringe a norma Frobenius dos tensores. |

### 4. Melhores Parâmetros Encontrados no Estudo
- **Learning Rate:** $0.0314$
- **Batch Size:** $256$
- **Camadas Ocultas:** $3$ camadas ($32 \to 64 \to 128$)
- **Dropout:** $0.216$
- **L2 Regularizer:** $1.46 \times 10^{-5}$
- **Val Loss Obtido:** $0.5467$

### 5. Onde está no Código?
- Arquivo: [`src/tuning.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/tuning.py#L12-L120)
- Classes e Funções: `OptunaPruningCallback` (L12-31), `create_mlp_objective` (L32-99) e `run_optuna_study` (L101-120).
- Material de Referência: Slide *"Deep Learning - parametrizacao 2026_2 (1).pdf"*.

---

## Tópico 8: Métricas de Avaliação e Análise de Custo-Efetividade Computacional

### 1. Por que Acurácia é Ilusória?
Em um cenário com $79\%$ de pontualidade e $21\%$ de atrasos:
- Um modelo "bobo" que sempre prevê classe 0 atinge **$78.8\%$ de acurácia**, mas tem **$0\%$ de utilidade prática**, pois não detecta nenhum atraso.
- Por isso, avaliamos métricas desenhadas para assimetria de classes:
  * **Precision (Atraso):** De todos os voos que o modelo disse que atrasariam, quantos realmente atrasaram? (Evita alarmes falsos).
  * **Recall (Atraso):** De todos os voos que atrasaram na realidade, quantos o modelo conseguiu capturar? (Evita surpresas operacionais).
  * **Macro F1-Score:** A média harmônica não-enviesada do F1 entre as duas classes:
    $$\text{Macro-F1} = \frac{F1_0 + F1_1}{2}$$
  * **AUC-ROC (Área sob a Curva ROC):** Capacidade intrínseca da rede de ranquear voos atrasados com probabilidades mais altas que voos pontuais, independente do limiar de corte de probabilidade.

### 2. Tabela Comparativa Consolidada no Conjunto de Teste Cego
| Modelo Avaliado | Macro F1-Score | AUC-ROC | Parâmetros Treináveis | Latência de Inferência (ms / 1k amostras) |
| :--- | :---: | :---: | :---: | :---: |
| **MLP V1 (Rasa Baseline)** | 0.4348 | 0.5464 | **2.433** | **119.97 ms** (Ultra-rápido) |
| **MLP V2 (Profunda Regularizada)** | **0.4936** | 0.4761 | 51.585 | 207.89 ms |
| **MLP V3 (Otimizada Optuna)** | 0.4503 | **0.5472** | 12.193 | 231.57 ms (4x menos parâmetros que V2) |
| **Recorrente LSTM (Sequencial)** | 0.4443 | 0.5223 | 25.921 | 316.30 ms (+52% de lentidão) |

### 3. Síntese do Critério de Custo vs. Efetividade
- A **MLP V2** é a melhor escolha quando a prioridade é o equilíbrio de detecção entre as classes (maior Macro-F1).
- A **MLP V3 (Optuna)** é a arquitetura mais eficiente em separabilidade (maior AUC-ROC) com apenas 12 mil parâmetros (ideal para dispositivos embarcados em quiosques aeroportuários ou mobile).
- A **LSTM** não se justificou técnica ou economicamente para este formato tabular de malha aérea.

---

## Tópico 9: Como Executar Novos Experimentos e Buscar Resultados Superiores

Você pode conduzir novos experimentos para testar hipóteses, expandir as tentativas do Optuna e buscar superar a marca de Macro-F1 da MLP V2.

### 1. Executando o Pipeline Completo via Terminal
No terminal PowerShell da raiz do repositório:
```powershell
# Execução padrão (5 trials do Optuna para verificação rápida)
python main.py
```

### 2. Expandindo o Estudo do Optuna para Mais Tentativas
Abra o arquivo [`main.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/main.py#L25) e altere a chamada de `run_pipeline`:
```python
# Aumentar trials de 5 para 30 ou 50 para exploração profunda:
run_pipeline(optuna_trials=30)
```
Ou execute o caderno interativo:
- Abra [`notebooks/03_optuna_experiments.ipynb`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/notebooks/03_optuna_experiments.ipynb).
- O caderno salva o progresso no banco de dados SQLite persistente em [`results/optuna_study.db`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/results/optuna_study.db), o que significa que novos trials continuam a partir dos anteriores!

### 3. Três Experimentos Práticos para Tentar Bater o Macro-F1

#### Experimento A: Ajuste do Limiar de Decisão (Threshold Tuning)
Por padrão, a saída sigmoide classifica como atraso se $\hat{y} \ge 0.50$. Em dados desbalanceados, se você variar esse limiar entre $0.30$ e $0.45$, o Recall de atrasos aumenta substancialmente sem sacrificar a precisão!
- **Como testar:** No arquivo [`src/evaluation.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/evaluation.py#L17), a função `evaluate_model(model, X_test, y_test, threshold=0.5)` aceita o argumento `threshold`.
- Teste com `threshold=0.38` ou `threshold=0.42` para observar o salto no Macro-F1.

#### Experimento B: Testar Ativação Swish ou GELU
No arquivo [`src/models.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/models.py#L41), substitua `relu` por `'swish'` (ativação suave auto-gated usada na EfficientNet):
```python
x = layers.Activation('swish', name=f"swish_{idx+1}")(x)
```
A função Swish permite que gradientes negativos pequenos fluam através da rede sem "matar" neurônios como ocorre com ReLUs inativas.

#### Experimento C: Testar Otimizador Adam com Learning Rate Scheduler
Adicionar um decaimento suave de taxa de aprendizado com `ReduceLROnPlateau`:
```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-5
)
```
Isso desacelera a taxa de aprendizado quando o modelo estabiliza, permitindo um refinamento mais preciso no mínimo da função de perda.

---

## Resumo dos Arquivos Principais do Projeto

| Arquivo | Função Principal no Projeto |
| :--- | :--- |
| [`src/data_pipeline.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/data_pipeline.py) | Ingestão, filtros, features cíclicas, split temporal estrito e ColumnTransformer anti-leakage. |
| [`src/models.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/models.py) | Fábrica de modelos Keras: MLP modular (BatchNorm/Dropout/L2), Recorrente (LSTM/GRU) e Autoencoders. |
| [`src/tuning.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/tuning.py) | Otimizador Bayesiano Optuna com MedianPruner, banco SQLite e função objetivo. |
| [`src/evaluation.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/src/evaluation.py) | Cálculo de métricas desbalanceadas (Macro-F1, ROC-AUC), curvas de aprendizado e latência. |
| [`main.py`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/main.py) | Script de orquestração ponta a ponta do pipeline experimental completo. |
| [`reports/relatorio_tecnico.html`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/reports/relatorio_tecnico.html) | Relatório acadêmico completo com fórmulas renderizadas via KaTeX e tabelas acadêmicas. |
| [`reports/slides_apresentacao.html`](file:///c:/Users/User/Documents/GitHub/deep_learning_trabalho_final/reports/slides_apresentacao.html) | Apresentação em slides interativos para a defesa do trabalho para a turma. |
