# SPEC-02: ESPECIFICAÇÃO DE MODELAGEM E ARQUITETURAS NEURAIS
## Trabalho Prático de Deep Learning — Voo Regular Ativo (VRA / ANAC)
### Disciplina: Deep Learning (2026/2) — Prof. Dr. Sérgio Canuto

---

### Identificação da Especificação
- **Identificador:** `SPEC-02`
- **Componente:** Fábrica de Modelos Neurais Keras / TensorFlow
- **Módulo de Código Alvo:** `src/models.py`
- **Status:** **Aprovado para Implementação**

---

## 1. Fundamentos Teóricos (Base RAG do Prof. Sérgio Canuto)

Todas as arquiteturas implementadas utilizam primitivas formalmente ensinadas nos slides da disciplina:

1. **Teorema da Aproximação Universal (Hornik, 1989 / Kolmogorov):**
   - Uma rede feedforward com uma única camada oculta e número finito de neurônios pode aproximar qualquer função contínua em subconjuntos compactos de $\mathbb{R}^n$, desde que dotada de função de ativação não-linear.
2. **Funções de Ativação:**
   - **$\text{ReLU}(z) = \max(0, z)$:** Padrão para camadas ocultas densas e convolucionais, acelerando o gradiente e mitigando a saturação.
   - **$\text{ELU}(z)$ / $\text{Leaky ReLU}$:** Alternativas para combater o problema dos "neurônios mortos" (*Dying ReLU*).
   - **$\sigma(z) = \frac{1}{1 + e^{-z}}$:** Utilizada exclusivamente na camada de saída para classificação binária.
3. **Inicialização de Pesos:**
   - **He Normal / He Uniform:** Inicialização mandatória para camadas com ativação $\text{ReLU}$ ou variantes ($\sigma = \sqrt{2 / n_{\text{in}}}$).
   - **Glorot / Xavier Uniform:** Inicialização para ativações simétricas ou $\text{Sigmoid}$.
4. **Regularização:**
   - **Dropout (Srivastava et al.):** Desativação aleatória de frações de neurônios durante o treinamento para prevenir coadaptações espúrias.
   - **Regularização $\text{L}_2$ (Weight Decay):** Penalização quadrática dos pesos na função de custo ($\frac{\lambda}{2} \|w\|^2$).
   - **Batch Normalization (Ioffe & Szegedy):** Normalização das ativações no minilote para suavizar a superfície de perda e permitir taxas de aprendizado maiores.

---

## 2. Frente Principal de Desenvolvimento: Multilayer Perceptrons (MLP - Frente 3.1)

A equipe avalia sistematicamente três variações promissoras de arquitetura para cumprir a Seção 3.1 do Edital:

### Variação 1: MLP Rasa Baseline (Capacidade Mínima)
- **Topologia:** Entrada $\to$ Densa(64, $\text{ReLU}$, He) $\to$ Densa(1, $\text{Sigmoid}$, Glorot).
- **Sem Regularização:** Sem Dropout, sem $\text{L}_2$, sem Batch Normalization.
- **Finalidade Científica:** Estabelecer o piso de desempenho e servir como diagnóstico de capacidade inicial.

### Variação 2: MLP Profunda Regularizada (Anti-Overfitting)
- **Topologia:**
  $$\text{Entrada} \to \text{Densa}(256) \to \text{BatchNorm} \to \text{Dropout}(0.3) \to \text{Densa}(128, \text{L}_2=10^{-4}) \to \text{BatchNorm} \to \text{Dropout}(0.2) \to \text{Densa}(64) \to \text{Densa}(1, \text{Sigmoid})$$
- **Finalidade Científica:** Avaliar a capacidade de generalização e a contenção do gap treino-validação em uma rede de maior profundidade.

### Variação 3: MLP Otimizada via Busca Bayesiana (Optuna)
- **Topologia:** Gerada dinamicamente pela melhor tentativa (*best trial*) do estudo Optuna (profundidade, largura, taxa de dropout e $\text{L}_2$ ótimos).
- **Finalidade Científica:** Comparar a parametrização empírica manual contra a busca automatizada bayesiana.

---

## 3. Frente Comparativa Experimental: Redes Recorrentes (RNN / LSTM / GRU - Frente 3.2)

Para o problema comparativo de séries temporais de atraso (Seção 3.2 do Edital), os dados são moldados em tensores tridimensionais:

$$\mathbf{X} \in \mathbb{R}^{\text{N\_amostras} \times T \times D}$$

Onde $T \in \{5, 10, 20\}$ representa a janela histórica de voos consecutivos da mesma rota/aeroporto.

### Comparação Tripla Obrigatória sob Condições Idênticas:
1. **SimpleRNN:**
   $$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
   *Objetivo de análise:* Demonstrar a vulnerabilidade ao desaparecimento do gradiente (*Vanishing Gradient*) em janelas longas ($T=20$).
2. **GRU (Gated Recurrent Unit):**
   $$z_t = \sigma(W_z x_t + U_z h_{t-1}), \quad r_t = \sigma(W_r x_t + U_r h_{t-1})$$
   $$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1})), \quad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
3. **LSTM (Long Short-Term Memory):**
   $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad \text{[Forget Gate]}$$
   $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i), \quad \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) \quad \text{[Input Gate]}$$
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{[Cell State Update]}$$
   $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o), \quad h_t = o_t \odot \tanh(c_t) \quad \text{[Output Gate]}$$

---

## 4. Frente 3.4: Autoencoders (Representação e Pré-Processamento)

Conforme os slides de Autoencoders do Prof. Sérgio Canuto:

### 4.1 Autoencoder Subcompleto com Espaço Latente 2D
- **Encoder:** $X \in \mathbb{R}^D \to \text{Densa}(64, \text{ReLU}) \to \text{Densa}(32, \text{ReLU}) \to z \in \mathbb{R}^2$
- **Decoder:** $z \in \mathbb{R}^2 \to \text{Densa}(32, \text{ReLU}) \to \text{Densa}(64, \text{ReLU}) \to \hat{X} \in \mathbb{R}^D$
- **Função de Perda:** $\mathcal{L}_{\text{rec}} = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$ (MSE)
- **Visualização 2D:** Plot bidimensional de $z = (z_1, z_2)$ colorindo as amostras por voo pontual vs. voo atrasado para verificar separabilidade manifold.

### 4.2 Denoising e Imputação
- **Denoising Autoencoder:** Injeção de perturbação gaussiana $\tilde{x} = x + \epsilon$, onde $\epsilon \sim \mathcal{N}(0, \sigma^2)$, com $\sigma = 0.1$, treinando a rede para reconstruir o vetor limpo $x$.
- **Alimentação da MLP:**
  $$\text{Vetor Latente } z \in \mathbb{R}^{d_{\text{latente}}} \longrightarrow \text{MLP Supervisionada} \longrightarrow \hat{y}$$
  Comparação de acurácia/F1 entre MLP com $z$ compactado vs. MLP treinada com $X$ original.

---

## 5. Implementação de Referência do Módulo `src/models.py`

```python
"""
Módulo de Fábrica de Modelos Neurais Keras / TensorFlow
"""
import tensorflow as tf
from tensorflow.keras import layers, regularizers

def build_mlp(input_dim: int,
              hidden_units=[128, 64],
              dropout_rate=0.2,
              l2_reg=1e-4,
              use_batch_norm=True,
              learning_rate=1e-3) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    
    for units in hidden_units:
        x = layers.Dense(
            units,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
            
    outputs = layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MLP_VRA")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_recurrent_classifier(input_shape: tuple,
                               cell_type='LSTM',
                               units=64,
                               dropout_rate=0.2,
                               learning_rate=1e-3) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    
    if cell_type.upper() == 'SIMPLERNN':
        rnn_layer = layers.SimpleRNN(units, return_sequences=False)
    elif cell_type.upper() == 'GRU':
        rnn_layer = layers.GRU(units, return_sequences=False)
    elif cell_type.upper() == 'LSTM':
        rnn_layer = layers.LSTM(units, return_sequences=False)
    else:
        raise ValueError(f"Célula desconhecida: {cell_type}")
        
    x = rnn_layer(inputs)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{cell_type}_VRA")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_autoencoder(input_dim: int, latent_dim=2, learning_rate=1e-3):
    # Encoder
    enc_inputs = layers.Input(shape=(input_dim,))
    e = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(enc_inputs)
    e = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(e)
    latent = layers.Dense(latent_dim, activation='linear', name='latent_space')(e)
    encoder = tf.keras.Model(inputs=enc_inputs, outputs=latent, name="Encoder")
    
    # Decoder
    dec_inputs = layers.Input(shape=(latent_dim,))
    d = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(dec_inputs)
    d = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(d)
    dec_outputs = layers.Dense(input_dim, activation='linear')(d)
    decoder = tf.keras.Model(inputs=dec_inputs, outputs=dec_outputs, name="Decoder")
    
    # Autoencoder completo
    ae_outputs = decoder(encoder(enc_inputs))
    autoencoder = tf.keras.Model(inputs=enc_inputs, outputs=ae_outputs, name="Autoencoder_VRA")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    
    return autoencoder, encoder, decoder
```

---

## 6. Critérios de Aceite da SPEC-02
- [ ] Todas as funções construtoras aceitam dimensões dinâmicas de entrada.
- [ ] O modelo compila sem erros com otimizador `Adam` e métricas adequadas.
- [ ] O método `model.summary()` exibe a contagem correta de parâmetros treináveis e não-treináveis.
- [ ] O modelo permite salvar e restaurar pesos no formato `.keras`.
