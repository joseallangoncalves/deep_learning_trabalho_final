"""
Módulo de Hiperparametrização Bayesiana com Optuna
Projeto: Voo Regular Ativo (VRA / ANAC) - Disciplina Deep Learning (2026/2)
Prof. Dr. Sérgio Canuto
"""
from typing import Dict, Any, Callable
import optuna
import tensorflow as tf
from tensorflow.keras import callbacks
from src.models import build_mlp

class OptunaPruningCallback(callbacks.Callback):
    """
    Callback Keras que reporta a métrica de validação ao Optuna e interrompe
    tentativas infrutíferas precocemente (MedianPruner).
    """
    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            message = f"Tentativa {self.trial.number} podada na época {epoch} com {self.monitor}={current_score:.4f}"
            raise optuna.TrialPruned(message)

def create_mlp_objective(
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    class_weights: Dict[int, float] = None, 
    max_epochs: int = 30
) -> Callable[[optuna.Trial], float]:
    """
    Gera a função objetivo do Optuna explorando o espaço hiperparamétrico justificado teoricamente:
    - learning_rate: escala logarítmica [1e-4, 1e-1]
    - batch_size: {32, 64, 128, 256}
    - n_layers: [1, 4]
    - units: {32, 64, 128, 256}
    - dropout_rate: [0.0, 0.5]
    - l2_reg: escala logarítmica [1e-6, 1e-2]
    """
    def objective(trial: optuna.Trial) -> float:
        # Amostragem dos hiperparâmetros
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
        
        # Arquitetura oculta da MLP
        hidden_units = []
        for i in range(n_layers):
            u = trial.suggest_categorical(f"units_layer_{i+1}", [32, 64, 128, 256])
            hidden_units.append(u)
            
        model = build_mlp(
            input_dim=X_train.shape[1],
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            use_batch_norm=True,
            learning_rate=lr
        )
        
        # Callbacks
        pruning_cb = OptunaPruningCallback(trial, monitor="val_loss")
        early_stop_cb = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
        
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=max_epochs,
                batch_size=batch_size,
                class_weight=class_weights,
                callbacks=[pruning_cb, early_stop_cb],
                verbose=0
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Falha de convergência: {e}")
            
        # A métrica é avaliada EXCLUSIVAMENTE na validação
        best_val_loss = float(min(history.history["val_loss"]))
        return best_val_loss

    return objective

def run_optuna_study(
    objective_func: Callable[[optuna.Trial], float],
    n_trials: int = 30,
    study_name: str = "vra_mlp_tuning",
    storage_path: str = "sqlite:///results/optuna_study.db"
) -> optuna.Study:
    """
    Executa a otimização bayesiana com persistência SQLite e MedianPruner.
    """
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(objective_func, n_trials=n_trials)
    return study
