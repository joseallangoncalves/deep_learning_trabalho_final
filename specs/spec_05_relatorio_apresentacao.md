# SPEC-05: ESPECIFICAÇÃO DO RELATÓRIO TÉCNICO, SLIDES E DEFESA ACADÊMICA
## Trabalho Prático de Deep Learning — Voo Regular Ativo (VRA / ANAC)
### Disciplina: Deep Learning (2026/2) — Prof. Dr. Sérgio Canuto

---

### Identificação da Especificação
- **Identificador:** `SPEC-05`
- **Componente:** Relatório Técnico Final, Slides de Apresentação (10 min) e Repositório GitHub
- **Artefatos Alvo:** `reports/relatorio_tecnico.pdf`, `reports/slides_apresentacao.pdf`, `README.md`
- **Status:** **Aprovado para Implementação**

---

## 1. Estrutura Obrigatória do Relatório Técnico (Edital — Seção 5)

O relatório deve conter entre 6 e 12 páginas, estruturado obrigatoriamente nas **6 seções formais**:

### Seção 1: Introdução e Objetivo do Problema
- **Contextualização:** Desafios econômicos e operacionais da pontualidade no transporte aéreo brasileiro.
- **Utilidade da Previsão:** Antecipação de gargalos de conectividade, alocação de equipes de solo e mitigação de custos de assistência aos passageiros.
- **Unidade de Observação:** O registro de um voo comercial individual no HOTRAN / VRA.
- **Definição da Variável Alvo:** Classificação binária de atraso de chegada $> 15$ minutos ($y \in \{0, 1\}$).

### Seção 2: Descrição da Fonte e Caracterização do Conjunto de Dados
- **Metadados:** Link oficial do Portal de Dados Abertos (VRA/ANAC) e período amostral selecionado.
- **Estatísticas Descritivas (EDA):** Tabela de médias, desvios e percentis de duração de voo e horários.
- **Tratamento de Nulos e Outliers:** Remoção de anomalias de digitação e voos cancelados sem dados operacionais.
- **Blindagem Anti-Leakage:** Prova formal de exclusão de variáveis concorrentes (`ChegadaReal`, `CodigoJustificativa`) e isolamento do `.fit()` no treinamento.

### Seção 3: Arquiteturas, Espaço de Busca e Protocolo de Hiperparametrização
- **Frente Selecionada:** Detalhamento topológico da arquitetura (camadas densas, convolucionais, recorrentes ou autoencoders).
- **Funções de Ativação e Regularização:** Justificativa da $\text{ReLU}$, $\text{Sigmoid}$, He Normal, Dropout e $\text{L}_2$.
- **Estudo Optuna:** Registro formal do espaço de busca, escala logarítmica da taxa de aprendizado, algoritmo TPE e podas efetuadas pelo `MedianPruner`.

### Seção 4: Resultados de Validação e Seleção dos Modelos Promissores
- **Análise das Tentativas:** Gráficos do Optuna (`plot_optimization_history`, `plot_param_importances`).
- **Comparação das 3 Variações:** Tabela comparativa na validação.
- **Diagnóstico das Curvas de Aprendizado:** Gráficos de perda e métricas treino vs. validação evidenciando a ausência de sobreajuste severo devido à regularização e ao Early Stopping.

### Seção 5: Avaliação Final no Conjunto de Teste e Análise de Custo vs. Efetividade
- **Desempenho Cego no Teste:** Tabela com Macro-F1, Precision, Recall, AUC-ROC e Matriz de Confusão Normalizada.
- **Análise Custo-Efetividade:** Comparação de número de parâmetros, tempo de treinamento por época e tempo de inferência para 1.000 amostras.
- **Discussão do Trade-Off:** Avaliação crítica se o ganho de Macro-F1 compensa o custo computacional em sistemas de tráfego aéreo em tempo real.

### Seção 6: Limitações e Conclusões
- **Fontes de Incerteza:** Variáveis meteorológicas (vento, tempestades), manutenção não programada e tráfego aéreo em rota não disponíveis no VRA básico.
- **Trabalhos Futuros:** Fusão com dados meteorológicos (METAR/TAF) e modelos híbridos CNN-LSTM.

---

## 2. Roteiro Cronometrado para Apresentação em Slides (10 Minutos)

O Edital estipula o limite estrito de **10 minutos** perante a banca:

| Bloco de Tempo | Slides | Título e Conteúdo Chave | Responsável / Foco da Fala |
| :---: | :---: | :--- | :--- |
| **0:00 - 2:00** | **Slide 1 & 2** | **Contexto & Variável Alvo:** Apresentação do VRA/ANAC, utilidade da predição de atraso crítico ($>15$ min) e impacto regulatório. | Foco na formulação matemática e clareza do problema de negócio. |
| **2:00 - 4:00** | **Slide 3 & 4** | **Dados & Pipeline Anti-Leakage:** Estatísticas do VRA, tratamento de desbalanceamento, features cíclicas e divisão cronológica estrita (sem shuffle). | Enfatizar o isolamento do `.fit()` exclusivo no treino. |
| **4:00 - 7:00** | **Slide 5 & 6** | **Modelagem & Busca Bayesiana (Optuna):** Arquitetura da rede, espaço de busca logarítmico, ação do `MedianPruner` e curvas de convergência (treino vs. validação). | Demonstrar domínio teórico de parametrização e controle de overfitting. |
| **7:00 - 9:00** | **Slide 7** | **Resultados no Teste & Matriz de Confusão:** Macro-F1, sensibilidade para capturar atrasos e métricas no teste cego. | Demonstrar rigor estatístico sem viés de acurácia simples. |
| **9:00 - 10:00** | **Slide 8** | **Custo Computacional & Conclusão:** Tabela de parâmetros e latência de inferência, trade-off de recursos e limitações operacionais. | Conclusão firme e encerramento no tempo limite exato. |

---

## 3. Checklist de Submissão e Repositório GitHub

- [ ] **Repositório Público ou com Acesso ao Professor:**
  - `README.md` estruturado com instruções claras de instalação:
    ```bash
    git clone <url_repositorio>
    cd deep_learning_trabalho_final
    python -m venv venv
    source venv/bin/activate  # ou venv\Scripts\activate no Windows
    pip install -r requirements.txt
    ```
- [ ] **Histórico de Commits:** Commits descritivos evidenciando a evolução da implementação e versionamento.
- [ ] **Arquivos Pesados Fora do Git:** Arquivo `.gitignore` configurado para ignorar arquivos CSV brutos volumosos em `data/raw/` e bases SQLite temporárias.
- [ ] **Slides e Relatório:** Inclusos na pasta `reports/` em formato PDF.
