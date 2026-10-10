"""
Módulo de Avaliação Experimental, Métricas e Custo Computacional
Projeto: Voo Regular Ativo (VRA / ANAC) - Disciplina Deep Learning (2026/2)
Prof. Dr. Sérgio Canuto
"""
from typing import Dict, Any, Tuple
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Avaliação no conjunto de teste cego reportando métricas não enviesadas para dados desbalanceados:
    Macro F1, Precision, Recall, AUC-ROC e tempo de inferência.
    """
    start_time = time.perf_counter()
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    inf_duration = time.perf_counter() - start_time
    
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'f1_atraso': float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'precision_atraso': float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'recall_atraso': float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_prob)),
        'inf_ms_per_1k': float((inf_duration / len(X_test)) * 1000.0 * 1000.0)
    }
    return metrics, y_pred, y_pred_prob

def plot_learning_curves(history, save_path: str = "results/learning_curves.png"):
    """
    Gera gráficos de Loss e Desempenho ao longo das épocas (diagnóstico de underfitting/overfitting).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Curvas de Perda (Loss)
    ax1.plot(history.history['loss'], label='Treino', linewidth=2, color='#1f77b4')
    ax1.plot(history.history['val_loss'], label='Validação', linewidth=2, linestyle='--', color='#ff7f0e')
    ax1.set_title('Convergência da Perda (Binary Cross-Entropy)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Curvas de Desempenho
    metric_key = 'auc' if 'auc' in history.history else 'accuracy'
    ax2.plot(history.history[metric_key], label=f'Treino ({metric_key.upper()})', linewidth=2, color='#2ca02c')
    ax2.plot(history.history[f'val_{metric_key}'], label=f'Validação ({metric_key.upper()})', linewidth=2, linestyle='--', color='#d62728')
    ax2.set_title(f'Desempenho no Treinamento ({metric_key.upper()})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel(metric_key.upper(), fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix_annotated(y_test: np.ndarray, y_pred: np.ndarray, save_path: str = "results/confusion_matrix.png"):
    """
    Plota a matriz de confusão com valores absolutos e percentuais normalizados por classe.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels = ['Pontual (<=15m)', 'Atrasado (>15m)']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=labels, yticklabels=labels, cbar=False,
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title('Matriz de Confusão Normalizada (Teste)', fontsize=12, fontweight='bold')
    plt.xlabel('Predição da Rede Neural', fontsize=11)
    plt.ylabel('Rótulo Real ANAC', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def benchmark_model_cost(model, sample_batch: np.ndarray, n_repetitions: int = 20) -> Dict[str, Any]:
    """
    Mede com precisão parâmetros treináveis e latência média de inferência.
    Compatível com Keras 2 e Keras 3.
    """
    trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables])) if model.trainable_variables else model.count_params()
    non_trainable_params = int(np.sum([np.prod(v.shape) for v in model.non_trainable_variables])) if model.non_trainable_variables else 0
    
    # Warm-up de inferência
    for _ in range(3):
        _ = model(sample_batch[:5], training=False)
        
    durations = []
    for _ in range(n_repetitions):
        t0 = time.perf_counter()
        _ = model(sample_batch, training=False)
        durations.append(time.perf_counter() - t0)
        
    mean_duration_ms = float(np.mean(durations)) * 1000.0
    latency_per_1k_ms = float((mean_duration_ms / len(sample_batch)) * 1000.0)
    
    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': trainable_params + non_trainable_params,
        'latency_per_1k_ms': latency_per_1k_ms
    }
