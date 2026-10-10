"""
Módulo de Construção de Modelos Neurais Keras / TensorFlow
Projeto: Voo Regular Ativo (VRA / ANAC) - Disciplina Deep Learning (2026/2)
Prof. Dr. Sérgio Canuto
"""
from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, regularizers

def build_mlp(
    input_dim: int,
    hidden_units: List[int] = [128, 64],
    dropout_rate: float = 0.2,
    l2_reg: float = 1e-4,
    use_batch_norm: bool = True,
    learning_rate: float = 1e-3,
    output_activation: str = 'sigmoid',
    loss: str = 'binary_crossentropy'
) -> tf.keras.Model:
    """
    Constrói MLP Densa conforme as diretrizes da Frente 3.1:
    - Inicialização He Normal para camadas ReLU.
    - Regularização L2 (Weight Decay) e Dropout.
    - Batch Normalization para estabilização de ativações internas.
    """
    inputs = layers.Input(shape=(input_dim,), name="input_features")
    x = inputs
    
    for idx, units in enumerate(hidden_units):
        reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
        x = layers.Dense(
            units, 
            kernel_initializer='he_normal',
            kernel_regularizer=reg,
            name=f"dense_{idx+1}_{units}"
        )(x)
        
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{idx+1}")(x)
            
        x = layers.Activation('relu', name=f"relu_{idx+1}")(x)
        
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dropout_{idx+1}")(x)
            
    outputs = layers.Dense(
        1, 
        activation=output_activation, 
        kernel_initializer='glorot_uniform',
        name="output_target"
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"MLP_VRA_{len(hidden_units)}layers")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_recurrent_classifier(
    input_shape: Tuple[int, int],
    cell_type: str = 'LSTM',
    units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """
    Constrói modelo recorrente para a Frente 3.2 comparativa:
    Permite alternar entre SimpleRNN, GRU e LSTM sob condições experimentais idênticas.
    """
    inputs = layers.Input(shape=input_shape, name="sequence_input")
    
    ct = cell_type.upper()
    if ct == 'SIMPLERNN':
        rnn_layer = layers.SimpleRNN(units, return_sequences=False, name="simple_rnn")
    elif ct == 'GRU':
        rnn_layer = layers.GRU(units, return_sequences=False, name="gru_cell")
    elif ct == 'LSTM':
        rnn_layer = layers.LSTM(units, return_sequences=False, name="lstm_cell")
    else:
        raise ValueError(f"Tipo de célula recorrente inválido: '{cell_type}'. Escolha SimpleRNN, GRU ou LSTM.")
        
    x = rnn_layer(inputs)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="rnn_dropout")(x)
        
    outputs = layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name="rnn_output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"Recurrent_{ct}_VRA")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_autoencoder(
    input_dim: int, 
    latent_dim: int = 2, 
    learning_rate: float = 1e-3
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """
    Constrói Autoencoder subcompleto com representação latente 2D para a Frente 3.4.
    Retorna a tripla (autoencoder_completo, encoder_extrator, decoder).
    """
    # Encoder
    enc_inputs = layers.Input(shape=(input_dim,), name="encoder_input")
    e = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(enc_inputs)
    e = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(e)
    latent = layers.Dense(latent_dim, activation='linear', name="bottleneck_2d")(e)
    encoder = tf.keras.Model(inputs=enc_inputs, outputs=latent, name="Encoder_VRA")
    
    # Decoder
    dec_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    d = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(dec_inputs)
    d = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(d)
    dec_outputs = layers.Dense(input_dim, activation='linear', name="reconstruction")(d)
    decoder = tf.keras.Model(inputs=dec_inputs, outputs=dec_outputs, name="Decoder_VRA")
    
    # Autoencoder completo
    ae_outputs = decoder(encoder(enc_inputs))
    autoencoder = tf.keras.Model(inputs=enc_inputs, outputs=ae_outputs, name="Autoencoder_VRA")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return autoencoder, encoder, decoder
