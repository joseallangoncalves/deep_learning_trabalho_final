# Roteiro Completo de Apresentação em Slides (10 Minutos)
## Trabalho Prático de Deep Learning — VRA / ANAC (2026/2)
**Docente:** Prof. Dr. Sérgio Canuto  

---

### Slide 1: Abertura e Identificação (0:00 - 1:00)
- **Título:** Previsão de Atrasos Operacionais no Voo Regular Ativo (VRA / ANAC)
- **Subtítulo:** Modelagem Preditiva com Multilayer Perceptrons (MLP) e Comparativo Recorrente (LSTM)
- **Disciplina:** Deep Learning (2026/2) — Prof. Dr. Sérgio Canuto (IFG)
- **Equipe:** Discentes da Disciplina
- **Frente Principal:** Frente 3.1: Multilayer Perceptrons
- **Frente Comparativa:** Frente 3.2: Redes Recorrentes (LSTM)

---

### Slide 2: Contexto, Utilidade e Formulação Matemática (1:00 - 2:00)
- **O Dataset VRA:** Dados abertos da ANAC catalogados no Portal Brasileiro de Dados Abertos.
- **Unidade de Observação:** O registro de um voo comercial regular individual.
- **Utilidade Prática:** Mitigar custos operacionais, otimizar conexões de passageiros e alocação de equipes de solo.
- **Formulação Matemática:**
  $$\text{AtrasoChegadaMin} = \frac{\text{DataHoraReal}_{\text{chegada}} - \text{DataHoraPrevista}_{\text{chegada}}}{60 \text{ segundos}}$$
  $$y = \begin{cases} 1, & \text{se } \text{AtrasoChegadaMin} > 15 \text{ min (Atraso Crítico ANAC)} \\ 0, & \text{se } \text{AtrasoChegadaMin} \le 15 \text{ min (Pontual/Adiantado)} \end{cases}$$

---

### Slide 3: Engenharia de Recursos e Blindagem Anti-Leakage (2:00 - 3:00)
- **Eliminação de Data Leakage:**
  - Exclusão mandatória de `ChegadaReal`, `CodigoJustificativa` e variáveis conhecidas apenas pós-operação.
  - `StandardScaler` e `OneHotEncoder` ajustados (`fit`) **estritamente nos 70% de Treinamento**.
- **Features Cíclicas de Horário e Calendário:**
  - Projeção senoidal e cossinoidal $(\sin, \cos)$ para hora de partida programada, dia da semana e mês.
  - Garante continuidade física entre 23:59 e 00:00.

---

### Slide 4: Particionamento Cronológico e Desbalanceamento (3:00 - 4:00)
- **Divisão Cronológica:**
  - Treino: $70\%$ (4.027 voos mais antigos).
  - Validação: $15\%$ (863 voos intermediários para Optuna e Early Stopping).
  - Teste: $15\%$ (863 voos mais recentes para avaliação cega).
  - **Sem Shuffle:** Previne previsão do passado utilizando o futuro.
- **Tratamento do Desbalanceamento:**
  - $78.8\%$ Pontuais vs. $21.2\%$ Atrasados.
  - Pesos de classe na perda: $w_0 = 0.637, w_1 = 2.325$.
  - Métrica de decisão: **Macro F1-Score** e AUC-ROC.

---

### Slide 5: Frente 3.1: Multilayer Perceptrons (MLP) (4:00 - 5:30)
- **Variação 1 (Rasa Baseline):** 1 camada oculta ($64$ neurônios), sem regularização. Piso de capacidade.
- **Variação 2 (Profunda Regularizada):** 3 camadas ($256 \to 128 \to 64$), Dropout $0.3$, Batch Normalization e $\text{L}_2 = 10^{-4}$.
- **Variação 3 (Otimizada Optuna):** Configuração piramidal ($32 \to 64 \to 128$), Dropout $0.21$, $\text{L}_2 = 1.4 \times 10^{-5}$ e Batch Normalization.
- **Inicializadores:** He Normal para ReLU e Glorot Uniform para Sigmoide.

---

### Slide 6: Otimização Bayesiana com Optuna & MedianPruner (5:30 - 7:00)
- **Espaço de Busca com Justificativa Teórica:**
  - $\text{Learning Rate} \in [10^{-4}, 10^{-1}]$ (escala logarítmica).
  - $\text{Batch Size} \in \{32, 64, 128, 256\}$.
  - $\text{Dropout} \in [0.0, 0.5]$ e $\text{L}_2 \in [10^{-6}, 10^{-2}]$.
- **Poda Inteligente com MedianPruner:**
  - Amostrador TPE (Tree-structured Parzen Estimator).
  - Interrupção precoce de trajetórias de validação inferiores à mediana histórica.
  - Economia de recursos computacionais.

---

### Slide 7: Avaliação Experimental Cega & Comparativo LSTM (7:00 - 9:00)
- **Tabela Geral no Conjunto de Teste Cego:**
  | Modelo Avaliado | Macro-F1 | AUC-ROC | Parâmetros | Latência (ms / 1k) |
  | :--- | :---: | :---: | :---: | :---: |
  | **MLP V1 (Rasa Baseline)** | 0.4348 | 0.5464 | 2.433 | **119.97 ms** |
  | **MLP V2 (Profunda Regularizada)** | **0.4936** | 0.4761 | 51.585 | 207.89 ms |
  | **MLP V3 (Otimizada Optuna)** | 0.4503 | **0.5472** | 12.193 | 231.57 ms |
  | **Recorrente LSTM (Sequencial $W=5$)** | 0.4443 | 0.5223 | 25.921 | 316.30 ms (+52%) |
- **Destaque:** MLP V2 obteve o maior equilíbrio entre classes (Macro-F1).

---

### Slide 8: Custo-Efetividade Computacional & Conclusões (9:00 - 10:00)
- **Veredito do Comparativo:**
  - A LSTM exigiu 52% mais latência de inferência (316 ms vs. 207 ms) sem ganho expressivo de Macro-F1 em relação à MLP densa.
  - Para sistemas de controle em tempo real, a MLP profunda regularizada provou ser a solução mais custo-efetiva e robusta.
- **Limitações:** Dados meteorológicos em rota (METAR) não presentes nos dados abertos básicos do VRA.
- **Trabalhos Futuros:** Fusão multimodal e modelos com mecanismo de atenção.
- **Agradecimento e Perguntas da Banca Examinadora.**
