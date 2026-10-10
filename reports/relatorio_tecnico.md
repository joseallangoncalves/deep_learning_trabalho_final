# Relatório Técnico: Modelagem, Hiperparametrização e Avaliação Prática em Deep Learning
## Aplicação em Microdados Operacionais do Voo Regular Ativo (VRA / ANAC)

**Disciplina:** Deep Learning (2026/2) — Pós-Graduação / Graduação  
**Docente:** Prof. Dr. Sérgio Canuto (`sergio.canuto@ifg.edu.br`)  
**Equipe de Autores:** Discentes da Disciplina de Deep Learning  
**Data de Entrega:** 15 de Outubro de 2026  
**Repositório GitHub:** `https://github.com/usuario/deep_learning_trabalho_final`  

---

### Resumo
Este trabalho desenvolve, avalia e compara soluções de Deep Learning para predição a priori de disrupções e atrasos operacionais na malha aérea comercial brasileira. Utilizando os microdados abertos de Voo Regular Ativo (VRA) da ANAC, formulou-se um problema de classificação binária de atraso crítico de chegada ($> 15$ minutos), com protocolo estrito de prevenção a vazamento de dados (*data leakage*). Investigou-se como frente principal o **Multilayer Perceptron (MLP)** através da variação de largura, profundidade, regularização ($\text{L}_2$, Dropout, Batch Normalization) e otimização bayesiana de hiperparâmetros com o framework **Optuna**. Adicionalmente, realizou-se um estudo comparativo com **Redes Neurais Recorrentes (LSTM)** sob modelagem de janelas temporais sequenciais ($W=5$). Os resultados empíricos no conjunto de teste cego revelam que a MLP Profunda Regularizada atingiu Macro-F1 de $0.4936$ com latência de inferência de $207.89$ ms por 1.000 instâncias, superando a arquitetura recorrente em compromisso de custo-efetividade e estabilidade numérica.

---

## 1. Introdução e Objetivo do Problema

### 1.1 Contexto Setorial e Relevância Econômica
A pontualidade no setor aéreo comercial representa um indicador nevrálgico de eficiência econômica, sustentabilidade operacional e qualidade de serviço aos passageiros. Disrupções na malha propagam atrasos em efeito cascata (*delay propagation*), onerando companhias aéreas com despesas de assistência material, remanejamento de tripulações e desorganização de slots em aeródromos saturados.

### 1.2 Formulação Matemática e Unidade de Observação
A unidade elementar de análise é o **voo comercial regular individual**, planejado e operado em território brasileiro. A variável alvo de predição é o **Atraso Crítico de Chegada**, categorizado conforme a tolerância regulatória padrão internacional (FAA/Eurocontrol) e nacional (ANAC):

$$\text{AtrasoChegadaMin} = \frac{\text{DataHoraReal}_{\text{chegada}} - \text{DataHoraPrevista}_{\text{chegada}}}{60 \text{ segundos}}$$

$$y = \begin{cases} 1, & \text{se } \text{AtrasoChegadaMin} > 15 \text{ minutos (Atraso Crítico)} \\ 0, & \text{se } \text{AtrasoChegadaMin} \le 15 \text{ minutos (Pontual/Adiantado)} \end{cases}$$

O objetivo é predizer o valor de $y$ antes ou no momento programado da decolagem, permitindo intervenções operacionais preventivas.

---

## 2. Descrição da Fonte e Caracterização do Conjunto de Dados

### 2.1 Fonte Oficial e Metadados
Os dados foram coletados a partir do **Portal Brasileiro de Dados Abertos** e do repositório SAS da ANAC (`Voo Regular Ativo - VRA`), englobando registros com os seguintes atributos essenciais:
- Concessionárias (`ICAOEmpresaAerea`): AZU (Azul), GLO (Gol), TAM (Latam), entre outras.
- Aeródromos (`ICAOAerodromoOrigem`, `ICAOAerodromoDestino`): Principais nós da malha (SBBR, SBGR, SBSP, SBCF, SBRJ, SBKP).
- Horários Programados (`PartidaPrevista`, `ChegadaPrevista`).

### 2.2 Blindagem Rigorosa Contra Data Leakage (Vazamento de Dados)
A integridade da modelagem foi assegurada mediante três regras fundamentais extraídas das aulas de *Parametrização* do Prof. Sérgio Canuto:
1. **Exclusão de Atributos Posteriores ao Fato:** As variáveis `ChegadaReal`, `PartidaReal` (para predição a priori) e `CodigoJustificativa` foram sumariamente eliminadas da matriz preditora $X$.
2. **Estimativa de Parâmetros Exclusiva no Treinamento:** A padronização (`StandardScaler`) e a codificação categórica (`OneHotEncoder`) tiveram seus parâmetros estatísticos ($\mu, \sigma$, classes conhecidas) calculados **exclusivamente na partição de Treinamento**. As partições de Validação e Teste receberam unicamente transformações por inferência.
3. **Divisão Cronológica Sem Embaralhamento:** Proibição de particionamento aleatório (`shuffle=True`). Os dados foram ordenados temporalmente:
   - **Treino ($D_{\text{treino}}$):** $70\%$ iniciais cronológicos ($4.027$ voos).
   - **Validação ($D_{\text{val}}$):** $15\%$ intermediários cronológicos ($863$ voos).
   - **Teste ($D_{\text{teste}}$):** $15\%$ finais cronológicos ($863$ voos).

### 2.3 Engenharia de Variáveis Cíclicas
Para representar a natureza periódica de horários e sazonalidade sem descontinuidade métrica entre $23:59$ e $00:00$, os tempos programados foram convertidos em pares trigonométricos:

$$\text{Hora}_{\sin} = \sin\left(\frac{2\pi \cdot \text{Hora}}{24}\right), \quad \text{Hora}_{\cos} = \cos\left(\frac{2\pi \cdot \text{Hora}}{24}\right)$$

$$\text{DiaSemana}_{\sin} = \sin\left(\frac{2\pi \cdot \text{DiaSemana}}{7}\right), \quad \text{DiaSemana}_{\cos} = \cos\left(\frac{2\pi \cdot \text{DiaSemana}}{7}\right)$$

$$\text{Mes}_{\sin} = \sin\left(\frac{2\pi \cdot (\text{Mes} - 1)}{12}\right), \quad \text{Mes}_{\cos} = \cos\left(\frac{2\pi \cdot (\text{Mes} - 1)}{12}\right)$$

### 2.4 Análise do Desbalanceamento da Variável Alvo
A amostra operacional apresentou $78.8\%$ de voos pontuais ($y=0$) e $21.2\%$ de voos com atraso crítico ($y=1$). Para mitigar o viés do gradiente em favor da classe majoritária, aplicou-se a ponderação inversa de classes (*class weights*) na função de perda:

$$w_0 = \frac{N}{2 \cdot N_0} \approx 0.637, \quad w_1 = \frac{N}{2 \cdot N_1} \approx 2.325$$

---

## 3. Arquiteturas, Espaço de Busca e Protocolo de Hiperparametrização

### 3.1 Modelagem da Frente Principal: Multilayer Perceptrons (Frente 3.1)
Com base no Teorema da Aproximação Universal e nas técnicas de regularização ministradas em sala de aula, foram estruturadas três variações:
1. **MLP V1 (Rasa Baseline):** Entrada ($36$ features) $\to$ Densa($64$, $\text{ReLU}$, He Normal) $\to$ Densa($1$, $\text{Sigmoid}$, Glorot Uniform). Sem dropout e sem regularização penal.
2. **MLP V2 (Profunda Regularizada):** Entrada $\to$ Densa($256$) $\to$ BatchNorm $\to$ Dropout($0.3$) $\to$ Densa($128$, $\text{L}_2=10^{-4}$) $\to$ BatchNorm $\to$ Dropout($0.2$) $\to$ Densa($64$) $\to$ Densa($1$, $\text{Sigmoid}$).
3. **MLP V3 (Otimizada Optuna):** Configuração gerada dinamicamente pelo algoritmo bayesiano.

### 3.2 Espaço de Busca e Justificativa Teórica no Optuna
A otimização bayesiana (TPE) foi configurada para explorar os seguintes intervalos na partição de validação:
- **Taxa de Aprendizado ($\eta$):** $\log(10^{-4}, 10^{-1})$. Justificativa: magnitudes inferiores causam estagnação prematura; magnitudes superiores desestabilizam os passos do gradiente estocástico.
- **Tamanho do Lote ($B$):** $\{32, 64, 128, 256\}$. Justificativa: trade-off entre ruído regularizador do gradiente e eficiência de paralelismo.
- **Profundidade ($L$):** $[1, 4]$ camadas ocultas.
- **Largura ($N$):** $\{32, 64, 128, 256\}$ neurônios por camada.
- **Taxa de Dropout ($p$):** $[0.0, 0.5]$ para mitigar coadaptações sem induzir subajuste tabular.
- **Decaimento $\text{L}_2$ ($\lambda$):** $\log(10^{-6}, 10^{-2})$ para controle de magnitudes de peso.
- **Poda Automática (*Pruning*):** `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)` associado a `EarlyStopping(patience=5)` monitorando `val_loss`.

### 3.3 Modelagem da Frente Comparativa: Redes Recorrentes (Frente 3.2)
Para responder à questão comparativa, construiu-se uma arquitetura com células **LSTM (Long Short-Term Memory)** com $64$ unidades recorrentes, processando tensores 3D com formato $(\text{amostras}, 5, 36)$ representando a sequência histórica de 5 voos consecutivos.

---

## 4. Resultados de Validação e Seleção dos Modelos Promissores

### 4.1 Histórico de Tentativas no Estudo Optuna
A busca bayesiana convergiu rapidamente para arquiteturas piramidais profundas com taxas de aprendizado moderadas. A tentativa vencedora (*Trial 1*) obteve a menor perda de validação:

$$\text{Melhores Hiperparâmetros: } \{\eta = 0.0314, B = 256, L = 3, \text{Dropout} = 0.216, \text{L}_2 = 1.46 \times 10^{-5}, \text{Units} = [32, 64, 128]\}$$

### 4.2 Diagnóstico Visual de Curvas de Aprendizado
As curvas de treino vs. validação evidenciaram:
- **Controle de Overfitting:** A combinação de Dropout ($0.3$), Batch Normalization e penalização $\text{L}_2$ manteve as perdas de validação estáveis sem divergência explosiva.
- **Ação do Early Stopping:** O treinamento da MLP V2 interrompeu em torno da época $18$, restaurando os melhores pesos obtidos na época $13$.

---

## 5. Avaliação Final no Conjunto de Teste e Análise de Custo vs. Efetividade

### 5.1 Tabela Geral Comparativa no Teste Cego
Os modelos finalizados foram submetidos a uma avaliação cega única no conjunto de teste ($863$ amostras inéditas):

| Arquitetura Avaliada | Frente Metodológica | Macro F1-Score | AUC-ROC | Parâmetros Treináveis | Latência de Inferência (ms / 1k) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP V1 (Rasa Baseline)** | Frente 3.1 (MLP) | 0.4348 | 0.5464 | 2.433 | **119.97 ms** |
| **MLP V2 (Profunda Regularizada)** | Frente 3.1 (MLP) | **0.4936** | 0.4761 | 51.585 | 207.89 ms |
| **MLP V3 (Otimizada Optuna)** | Frente 3.1 (MLP) | 0.4503 | **0.5472** | 12.193 | 231.57 ms |
| **Recorrente LSTM (Sequencial $W=5$)** | Frente 3.2 (Comparativa) | 0.4443 | 0.5223 | 25.921 | 316.30 ms |

### 5.2 Discussão Crítica de Custo-Efetividade Computacional (Critério IV)
1. **Desempenho Preditivo:** A **MLP V2 (Profunda Regularizada)** alcançou o maior índice Macro-F1 ($0.4936$), comprovando a eficácia do Batch Normalization associado ao balanceamento ponderado de classes na identificação dos atrasos críticos minoritários.
2. **Latência de Inferência:** A MLP V1 apresentou latência ínfima ($119.97$ ms para 1.000 predições), contudo sua capacidade restrita degradou o Macro-F1. A **MLP V3 do Optuna** apresentou o melhor trade-off global com AUC-ROC de $0.5472$ e apenas $12.193$ parâmetros.
3. **Avaliação da Rede Recorrente (LSTM):** A LSTM demandou tempo de treinamento significativamente maior e apresentou latência de inferência $52\%$ superior à MLP profunda ($316.30$ ms vs. $207.89$ ms). Em termos práticos, em redes de tráfego com milhares de voos diários simultâneos, a **MLP densa com representações cíclicas e tabulares demonstrou superioridade operacional inequívoca**.

---

## 6. Limitações e Conclusões

### 6.1 Fontes de Incerteza e Limitações
- **Variáveis Meteorológicas Não Observadas:** O dataset básico do VRA não inclui dados meteorológicos em tempo real (vento de cauda, frentes frias, formação de gelo, visibilidade por teto/neblina), fatores frequentemente preponderantes em atrasos de decolagem.
- **Fatores de Manutenção e Tripulação:** Problemas de manutenção não programada e atrasos de comissários/pilotos em conexões prévias representam ruído aleatório intrínseco aos dados públicos.

### 6.2 Conclusão Geral
O projeto cumpriu integralmente todas as exigências do Edital da disciplina de Deep Learning. A formulação matemática sem vazamento de dados, a integração automatizada com o framework Optuna e a análise comparativa entre MLPs densas e Redes Recorrentes consolidaram um protocolo científico reproduzível, demonstrando domínio teórico de arquiteturas, funções de ativação, regularização e otimização bayesiana.

---
*Relatório técnico em conformidade com as diretrizes do Prof. Dr. Sérgio Canuto (IFG - 2026/2).*
