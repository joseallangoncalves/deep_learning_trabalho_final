# SPEC-03: ESPECIFICAÇÃO DE HIPERPARAMETRIZAÇÃO BAYESIANA COM OPTUNA
## Trabalho Prático de Deep Learning — Voo Regular Ativo (VRA / ANAC)
### Disciplina: Deep Learning (2026/2) — Prof. Dr. Sérgio Canuto

---

### Identificação da Especificação
- **Identificador:** `SPEC-03`
- **Componente:** Otimização Bayesiana de Hiperparâmetros, Callbacks e Pruning
- **Módulo de Código Alvo:** `src/tuning.py` e `notebooks/03_optuna_experiments.ipynb`
- **Status:** **Aprovado para Implementação**

---

## 1. Fundamentação Teórica da Busca Bayesiana (Aulas do Prof. Sérgio Canuto)

Conforme os slides de **Lidando com (Hiper)parâmetros**:
1. **Limitações do Grid Search e Random Search:**
   - O *Grid Search* sofre da maldição da dimensionalidade ($O(k^d)$ combinações) e explora intensamente combinações irrelevantes.
   - O *Random Search* amostra uniformemente sem memória de tentativas passadas.
2. **Otimização Bayesiana (TPE - Tree-structured Parzen Estimator):**
   - Modela probabilisticamente a função objetivo desconhecida $f(\theta)$ construindo uma distribuição a posteriori das configurações que maximizam o ganho esperado (*Expected Improvement*).
   - Equilibra *Exploração* (regiões incertas do espaço de busca) e *Explotação* (regiões conhecidas de alto desempenho).
3. **Poda Automática de Tentativas (*Pruning*):**
   - Tentativas com trajetórias de perda intermediárias estatisticamente inferiores à mediana histórica são interrompidas precocemente (`MedianPruner`), economizando horas de computação de GPU/CPU.

---

## 2. Espaço de Busca e Justificativas Formais do Edital (Seção 2.4)

| Hiperparâmetro | Distribuição Amostral | Intervalo / Valores | Justificativa Teórica Obrigatória |
| :--- | :--- | :--- | :--- |
| `learning_rate` ($\eta$) | `suggest_float(..., log=True)` | $[10^{-4}, 10^{-1}]$ | A magnitude da taxa de aprendizado altera exponencialmente o passo no gradiente descendente. Valores $> 0.1$ causam explosão/divergência numérica; valores $< 10^{-4}$ causam estagnação prematura. |
| `batch_size` ($B$) | `suggest_categorical(...)` | $\{32, 64, 128, 256\}$ | Lotes pequenos introduzem ruído estocástico que atua como regularizador implícito; lotes grandes aceleram o paralelismo do hardware mas exigem taxas de aprendizado ajustadas. |
| `n_layers` ($L$) | `suggest_int(...)` | $[1, 4]$ | Avalia se a complexidade da malha aérea requer representações hierárquicas profundas ou se redes rasas saturam a capacidade sem incorrer em sobreajuste. |
| `units` | `suggest_categorical(...)` | $\{32, 64, 128, 256\}$ | Define a largura da camada e o número de hiperplanos de corte no espaço de características. |
| `dropout_rate` ($p$) | `suggest_float(...)` | $[0.0, 0.5]$ | Varia de ausência de dropout ($0.0$) até atenuação moderada ($0.5$). Valores $> 0.5$ induzem subajuste severo em dados tabulares. |
| `l2_reg` ($\lambda$) | `suggest_float(..., log=True)` | $[10^{-6}, 10^{-2}]$ | Penalidade de norma quadrática nos pesos para restringir magnitudes extremas e suavizar a superfície de decisão. |

---

## 3. Protocolo de Execução do Estudo Optuna

1. **Direção da Otimização:** Minimizar a perda de validação (`val_loss`) ou maximizar o Macro F1 de validação (`val_macro_f1`).
2. **Amostrador (*Sampler*):** `optuna.samplers.TPESampler(seed=42)`.
3. **Poda (*Pruner*):** `optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)`.
4. **Número de Tentativas (*Trials*):** Mínimo de $25$ a $30$ tentativas registradas.
5. **Persistência do Estudo:** Banco de dados SQLite local (`sqlite:///results/optuna_study.db`), permitindo retomada do estudo em caso de falha de hardware.

---

## 4. Implementação de Referência do Módulo `src/tuning.py`

```python
"""
Módulo de Hiperparametrização Bayesiana com Optuna e Keras
"""
import optuna
import tensorflow as tf
from tensorflow.keras import callbacks
from src.models import build_mlp

class OptunaPruningCallback(callbacks.Callback):
    """Callback Keras que reporta a métrica de validação ao Optuna e dispara a poda."""
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
            message = f"Tentativa podada na época {epoch} com {self.monitor}={current_score:.4f}"
            raise optuna.TrialPruned(message)

def create_objective(X_train, y_train, X_val, y_val, class_weights=None, max_epochs=30):
    def objective(trial: optuna.Trial) -> float:
        # 1. Sugestão de Hiperparâmetros
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
        
        # Gerar topologia piramidal ou constante
        hidden_units = []
        for i in range(n_layers):
            units = trial.suggest_categorical(f"units_l{i}", [32, 64, 128, 256])
            hidden_units.append(units)
            
        # 2. Construção do Modelo
        model = build_mlp(
            input_dim=X_train.shape[1],
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            use_batch_norm=True,
            learning_rate=lr
        )
        
        # 3. Callbacks de Treino
        pruning_cb = OptunaPruningCallback(trial, monitor="val_loss")
        early_stopping_cb = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
        
        # 4. Ajuste no Treino e Validação
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=max_epochs,
                batch_size=batch_size,
                class_weight=class_weights,
                callbacks=[pruning_cb, early_stopping_cb],
                verbose=0
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            raise optuna.TrialPruned(f"Erro durante treino: {e}")
            
        best_val_loss = min(history.history["val_loss"])
        return best_val_loss

    return objective

def execute_study(objective_func, n_trials=30, storage_uri="sqlite:///results/optuna.db", study_name="vra_tuning"):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(objective_func, n_trials=n_trials)
    return study
```

---

## 5. Critérios de Aceite da SPEC-03
- [ ] O estudo Optuna executa no mínimo $25$ tentativas sem travamentos.
- [ ] O `MedianPruner` interrompe tentativas com desempenho degradado.
- [ ] O melhor conjunto de parâmetros (`study.best_params`) e a métrica de validação (`study.best_value`) são gravados e exportados para CSV/JSON.
- [ ] Gráficos de importância de hiperparâmetros gerados via `optuna.visualization.plot_param_importances(study)`.
