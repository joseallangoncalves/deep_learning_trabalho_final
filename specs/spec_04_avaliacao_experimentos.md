# SPEC-04: ESPECIFICAÇÃO DE AVALIAÇÃO EXPERIMENTAL E CUSTO COMPUTACIONAL
## Trabalho Prático de Deep Learning — Voo Regular Ativo (VRA / ANAC)
### Disciplina: Deep Learning (2026/2) — Prof. Dr. Sérgio Canuto

---

### Identificação da Especificação
- **Identificador:** `SPEC-04`
- **Componente:** Métricas de Avaliação, Análise de Overfitting e Benchmarking de Custo Computacional
- **Módulo de Código Alvo:** `src/evaluation.py` e `notebooks/04_final_evaluation.ipynb`
- **Status:** **Aprovado para Implementação**

---

## 1. Protocolo Científico de Avaliação no Teste Cego

Em conformidade rigorosa com a Seção 2.5 do Edital:
1. **Acesso Único ao Conjunto de Teste ($D_{\text{teste}}$):**
   - Os dados de teste **nunca** participam do ajuste de pesos nem da busca de hiperparâmetros com o Optuna.
   - Cada modelo arquitetural vencedor é avaliado uma única vez no teste para mensurar sua verdadeira capacidade de generalização.
2. **Tratamento do Desbalanceamento de Classes do VRA:**
   - Na malha aérea brasileira, aproximadamente $80\%$ a $85\%$ dos voos operam dentro da tolerância de pontualidade, enquanto $15\%$ a $20\%$ sofrem atrasos críticos ($> 15$ min).
   - **A Acurácia Simples é Proibida como Métrica Isolada**, pois um classificador ingênuo que sempre previsse "voo pontual" atingiria $>80\%$ de acurácia com utilidade nula.

---

## 2. Métricas de Desempenho Exigidas

### 2.1 Métricas de Classificação Binária

$$\text{Precisão (Precision)} = \frac{\text{VP}}{\text{VP} + \text{FP}}$$

$$\text{Revocação (Recall)} = \frac{\text{VP}}{\text{VP} + \text{FN}}$$

$$\text{F1-Score da Classe } c = 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}$$

$$\mathbf{\text{Macro F1-Score}} = \frac{\text{F1}_{\text{classe 0 (Pontual)}} + \text{F1}_{\text{classe 1 (Atrasado)}}}{2}$$

$$\text{AUC-ROC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

### 2.2 Diagnóstico Visual de Curvas de Aprendizado (Loss e Métricas por Época)
Os gráficos gerados devem permitir discutir na apresentação e no relatório:
- **Subajuste (Underfitting):** Perda de treino e validação estagnadas em patamar alto.
- **Sobreajuste (Overfitting):** Divergência entre perda de treino decrescente e perda de validação crescente.
- **Ponto de Parada Ótimo (Early Stopping):** Época com menor perda de validação com pesos restaurados.

---

## 3. Benchmarking de Custo-Efetividade Computacional (Seção 2.5 e Critério IV)

O Edital exige explicitamente relacionar desempenho preditivo com custo computacional. A tabela a seguir deve ser gerada automaticamente para os modelos comparados:

| Modelo Avaliado | Parâmetros Treináveis | Tempo de Treino / Época (s) | Tempo de Inferência (ms / 1k amostras) | Macro F1 (Teste) | AUC-ROC (Teste) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Variação 1 (MLP Rasa)** | *~15.000* | *~1.2 s* | *~8 ms* | *Ex: 0.72* | *Ex: 0.79* |
| **Variação 2 (MLP Profunda Reg)**| *~95.000* | *~2.8 s* | *~12 ms* | *Ex: 0.77* | *Ex: 0.84* |
| **Variação 3 (MLP Optuna)** | *~60.000* | *~2.1 s* | *~10 ms* | *Ex: 0.79* | *Ex: 0.86* |

---

## 4. Implementação de Referência do Módulo `src/evaluation.py`

```python
"""
Módulo de Avaliação Experimental, Métricas e Custo Computacional
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)

def evaluate_test_set(model, X_test, y_test, threshold=0.5) -> dict:
    start_inf = time.perf_counter()
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    inf_time_total = time.perf_counter() - start_inf
    
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'f1_atraso': f1_score(y_test, y_pred, pos_label=1),
        'precision_atraso': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'recall_atraso': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_prob),
        'inference_time_ms_per_1k': (inf_time_total / len(X_test)) * 1000 * 1000
    }
    return metrics, y_pred, y_pred_prob

def plot_learning_curves(history, save_path="results/learning_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Curvas de Perda (Loss)
    ax1.plot(history.history['loss'], label='Treino', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validação', linewidth=2, linestyle='--')
    ax1.set_title('Evolução da Perda (Binary Cross-Entropy)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Curvas de Acurácia ou AUC
    metric_name = 'auc' if 'auc' in history.history else 'accuracy'
    ax2.plot(history.history[metric_name], label=f'Treino ({metric_name})', linewidth=2)
    ax2.plot(history.history[f'val_{metric_name}'], label=f'Validação ({metric_name})', linewidth=2, linestyle='--')
    ax2.set_title(f'Evolução de Desempenho ({metric_name.upper()})')
    ax2.set_xlabel('Época')
    ax2.set_ylabel(metric_name.upper())
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix_annotated(y_test, y_pred, save_path="results/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=['Pontual (<=15m)', 'Atrasado (>15m)'],
        yticklabels=['Pontual (<=15m)', 'Atrasado (>15m)'],
        cbar=False, ax=ax
    )
    ax.set_title('Matriz de Confusão Normalizada (Teste)')
    ax.set_xlabel('Predição do Modelo')
    ax.set_ylabel('Rótulo Real ANAC')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def benchmark_model_cost(model, sample_input, n_repetitions=20) -> dict:
    trainable_params = int(np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables]))
    non_trainable_params = int(np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_variables]))
    
    # Aquecimento de inferência (Warm-up)
    for _ in range(3):
        _ = model(sample_input[:10])
        
    times = []
    for _ in range(n_repetitions):
        t0 = time.perf_counter()
        _ = model(sample_input, training=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        
    avg_inf_batch_ms = (np.mean(times)) * 1000
    inf_per_1k_ms = (avg_inf_batch_ms / len(sample_input)) * 1000
    
    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': trainable_params + non_trainable_params,
        'inf_per_1k_ms': inf_per_1k_ms
    }
```

---

## 5. Critérios de Aceite da SPEC-04
- [ ] O script gera relatório de métricas no teste cego (`classification_report` completo).
- [ ] Gráficos de curvas de aprendizado salvos e prontos para inserção no relatório e nos slides.
- [ ] Tabela de custo computacional relacionando parâmetros e tempos de inferência exportada em Markdown ou CSV.
