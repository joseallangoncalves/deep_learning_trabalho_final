"""
Pipeline Integrado de Execução Científica — VRA / ANAC
Disciplina Deep Learning (2026/2) — Prof. Dr. Sérgio Canuto
Executa o fluxo completo: Carga -> Split Anti-Leakage -> MLP (Var 1, 2, 3) -> Comparativo RNN -> Avaliação Teste.
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_pipeline import (
    load_vra_dataset, clean_and_prepare_vra,
    split_chronological, build_and_fit_preprocessor,
    transform_splits, create_recurrent_sequences
)
from src.models import build_mlp, build_recurrent_classifier
from src.tuning import create_mlp_objective, run_optuna_study
from src.evaluation import (
    evaluate_model, plot_learning_curves,
    plot_confusion_matrix_annotated, benchmark_model_cost
)
from src.generate_sample_vra import generate_sample_vra_dataset

def run_pipeline(data_path: str = "data/raw/vra_amostra_2023.csv", run_optuna: bool = True, optuna_trials: int = 5):
    print("=" * 70)
    print("INICIANDO EXECUÇÃO DO PROJETO DE DEEP LEARNING (VRA / ANAC)")
    print("Prof. Dr. Sérgio Canuto - Edital 2026/2")
    print("=" * 70)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    # 1. Obtenção ou Geração da Base VRA
    if not os.path.exists(data_path):
        print(f"\n[Fase 1] Base não encontrada em '{data_path}'. Gerando amostra representativa...")
        generate_sample_vra_dataset(data_path, n_samples=6000)
        
    print(f"\n[Fase 1] Carregando microdados de {data_path}...")
    raw_df = load_vra_dataset(data_path)
    print(f"Registros brutos carregados: {len(raw_df)}")
    
    # 2. Pré-processamento e Engenharia de Features sem Leakage
    print("\n[Fase 2] Higienizando dados e construindo variáveis cíclicas...")
    clean_df = clean_and_prepare_vra(raw_df, delay_threshold_min=15)
    print(f"Registros válidos (REALIZADO): {len(clean_df)}")
    print(f"Distribuição do alvo 'target_bin':")
    print(clean_df['target_bin'].value_counts(normalize=True).apply(lambda x: f"{x*100:.1f}%").to_dict())
    
    # 3. Particionamento Cronológico Estrito (70% Treino, 15% Validação, 15% Teste)
    print("\n[Fase 3] Realizando partição cronológica ordenada (sem shuffle)...")
    train_df, val_df, test_df = split_chronological(clean_df, train_ratio=0.70, val_ratio=0.15)
    print(f"Treinamento: {len(train_df)} amostras")
    print(f"Validação:   {len(val_df)} amostras")
    print(f"Teste:       {len(test_df)} amostras")
    
    num_cols = ['TempoVooPrevistoMin', 'hora_sin', 'hora_cos', 'dia_semana_sin', 'dia_semana_cos', 'mes_sin', 'mes_cos']
    cat_cols = ['ICAOEmpresaAerea', 'CodigoTipoLinha', 'ICAOAerodromoOrigem', 'ICAOAerodromoDestino']
    feature_cols = num_cols + cat_cols
    
    # 4. Ajuste do ColumnTransformer EXCLUSIVAMENTE no Treino
    print("\n[Fase 4] Ajustando ColumnTransformer (StandardScaler + OneHotEncoder) no treino...")
    preprocessor = build_and_fit_preprocessor(train_df, num_cols, cat_cols)
    X_train, y_train, X_val, y_val, X_test, y_test = transform_splits(
        preprocessor, train_df, val_df, test_df, feature_cols, 'target_bin'
    )
    print(f"Dimensão final do vetor de entrada X: {X_train.shape[1]} atributos")
    
    # Ponderação de classes (desbalanceamento)
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    class_weights = {
        0: float(len(y_train) / (2.0 * n_neg)),
        1: float(len(y_train) / (2.0 * n_pos))
    }
    print(f"Pesos de classe balanceados: {class_weights}")
    
    results_summary = []
    
    # =========================================================================
    # FRENTE 3.1: MULTILAYER PERCEPTRONS (ARQUITETURA PRINCIPAL)
    # =========================================================================
    print("\n" + "=" * 50)
    print("TREINAMENTO: FRENTE 3.1 - MULTILAYER PERCEPTRONS (MLP)")
    print("=" * 50)
    
    # Variação 1: MLP Rasa Baseline
    print("\n--> Treinando Variação 1: MLP Rasa Baseline (1 Camada 64)...")
    mlp_v1 = build_mlp(input_dim=X_train.shape[1], hidden_units=[64], dropout_rate=0.0, l2_reg=0.0, use_batch_norm=False, learning_rate=1e-3)
    mlp_v1.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=15, batch_size=128, class_weight=class_weights, verbose=0
    )
    metrics_v1, _, _ = evaluate_model(mlp_v1, X_test, y_test)
    cost_v1 = benchmark_model_cost(mlp_v1, X_test[:100])
    results_summary.append({
        'Modelo': 'MLP V1 (Rasa Baseline)',
        'Macro-F1': metrics_v1['macro_f1'],
        'AUC-ROC': metrics_v1['roc_auc'],
        'Parâmetros': cost_v1['trainable_params'],
        'Latência (ms / 1k)': metrics_v1['inf_ms_per_1k']
    })
    
    # Variação 2: MLP Profunda Regularizada
    print("--> Treinando Variação 2: MLP Profunda Regularizada (256-128-64 + Dropout + BatchNorm)...")
    mlp_v2 = build_mlp(input_dim=X_train.shape[1], hidden_units=[256, 128, 64], dropout_rate=0.3, l2_reg=1e-4, use_batch_norm=True, learning_rate=1e-3)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history_v2 = mlp_v2.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=25, batch_size=128, class_weight=class_weights, callbacks=[early_stop], verbose=0
    )
    metrics_v2, y_pred_v2, _ = evaluate_model(mlp_v2, X_test, y_test)
    cost_v2 = benchmark_model_cost(mlp_v2, X_test[:100])
    results_summary.append({
        'Modelo': 'MLP V2 (Profunda Regularizada)',
        'Macro-F1': metrics_v2['macro_f1'],
        'AUC-ROC': metrics_v2['roc_auc'],
        'Parâmetros': cost_v2['trainable_params'],
        'Latência (ms / 1k)': metrics_v2['inf_ms_per_1k']
    })
    plot_learning_curves(history_v2, save_path="results/learning_curves_mlp_v2.png")
    plot_confusion_matrix_annotated(y_test, y_pred_v2, save_path="results/confusion_matrix_mlp_v2.png")
    
    # Variação 3: MLP Otimizada via Optuna
    if run_optuna:
        print(f"\n--> Executando Busca Bayesiana com Optuna (Variação 3, {optuna_trials} tentativas)...")
        opt_obj = create_mlp_objective(X_train, y_train, X_val, y_val, class_weights=class_weights, max_epochs=15)
        study = run_optuna_study(opt_obj, n_trials=optuna_trials, study_name="vra_mlp_study")
        print(f"Melhores Parâmetros Optuna: {study.best_params}")
        
        # Treinar modelo final com os parâmetros do best_trial
        best_n_layers = study.best_params.get('n_layers', 2)
        best_units = [study.best_params.get(f"units_layer_{i+1}", 128) for i in range(best_n_layers)]
        mlp_v3 = build_mlp(
            input_dim=X_train.shape[1],
            hidden_units=best_units,
            dropout_rate=study.best_params.get('dropout_rate', 0.2),
            l2_reg=study.best_params.get('l2_reg', 1e-4),
            use_batch_norm=True,
            learning_rate=study.best_params.get('learning_rate', 1e-3)
        )
        mlp_v3.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=25, batch_size=study.best_params.get('batch_size', 128),
            class_weight=class_weights, callbacks=[early_stop], verbose=0
        )
        metrics_v3, _, _ = evaluate_model(mlp_v3, X_test, y_test)
        cost_v3 = benchmark_model_cost(mlp_v3, X_test[:100])
        results_summary.append({
            'Modelo': 'MLP V3 (Otimizada Optuna)',
            'Macro-F1': metrics_v3['macro_f1'],
            'AUC-ROC': metrics_v3['roc_auc'],
            'Parâmetros': cost_v3['trainable_params'],
            'Latência (ms / 1k)': metrics_v3['inf_ms_per_1k']
        })
        
    # =========================================================================
    # FRENTE 3.2: REDES RECORRENTES (ESTUDO COMPARATIVO TEMPORAL)
    # =========================================================================
    print("\n" + "=" * 50)
    print("TREINAMENTO: FRENTE 3.2 - REDES RECORRENTES (COMPARATIVO)")
    print("=" * 50)
    time_steps = 5
    print(f"Criando sequências 3D temporais com janela W = {time_steps} voos...")
    X_train_seq, y_train_seq = create_recurrent_sequences(X_train, y_train, time_steps=time_steps)
    X_val_seq, y_val_seq = create_recurrent_sequences(X_val, y_val, time_steps=time_steps)
    X_test_seq, y_test_seq = create_recurrent_sequences(X_test, y_test, time_steps=time_steps)
    
    print(f"Tensor 3D de Treino: {X_train_seq.shape}")
    
    # LSTM
    print("--> Treinando Modelo Comparativo LSTM (64 unidades)...")
    lstm_model = build_recurrent_classifier(input_shape=(time_steps, X_train.shape[1]), cell_type='LSTM', units=64, learning_rate=1e-3)
    lstm_model.fit(
        X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq),
        epochs=15, batch_size=128, verbose=0
    )
    metrics_lstm, _, _ = evaluate_model(lstm_model, X_test_seq, y_test_seq)
    cost_lstm = benchmark_model_cost(lstm_model, X_test_seq[:100])
    results_summary.append({
        'Modelo': 'Recorrente LSTM (Comparativo)',
        'Macro-F1': metrics_lstm['macro_f1'],
        'AUC-ROC': metrics_lstm['roc_auc'],
        'Parâmetros': cost_lstm['trainable_params'],
        'Latência (ms / 1k)': metrics_lstm['inf_ms_per_1k']
    })
    
    # =========================================================================
    # CONSOLIDAÇÃO FINAL E EXPORTAÇÃO DOS RESULTADOS
    # =========================================================================
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv("results/comparativo_modelos.csv", index=False)
    
    print("\n" + "=" * 70)
    print("TABELA COMPARATIVA FINAL NO CONJUNTO DE TESTE:")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("\n[OK] Execução concluída! Gráficos e tabelas salvos no diretório 'results/'.")

if __name__ == '__main__':
    run_pipeline()
