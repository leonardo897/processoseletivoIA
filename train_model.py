import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random

# Configuração de seeds para reprodutibilidade
def set_seeds(seed=42):
    """Define todas as seeds necessárias para reprodutibilidade"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Para TensorFlow garantir operações determinísticas
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

def train_model():
    # Define seed global
    SEED = 42
    set_seeds(SEED)
    print(f"Seeds configuradas com valor: {SEED}")
    
    print("Carregando dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Pré-processamento: normalização e reshape para adicionar canal
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Adiciona dimensão do canal (grayscale)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    print(f"Shape dos dados de treino: {x_train.shape}")
    print(f"Shape dos dados de teste: {x_test.shape}")

    # Construção da CNN (arquitetura simples para Edge AI)
    # Inicializadores com seed para reprodutibilidade
    kernel_init = keras.initializers.GlorotUniform(seed=SEED)
    bias_init = keras.initializers.Zeros()
    
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(8, kernel_size=(3, 3), activation="relu", 
                     kernel_initializer=kernel_init, bias_initializer=bias_init),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu",
                     kernel_initializer=kernel_init, bias_initializer=bias_init),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu",
                    kernel_initializer=kernel_init, bias_initializer=bias_init),
        layers.Dense(10, activation="softmax",
                    kernel_initializer=kernel_init, bias_initializer=bias_init)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    model.summary()

    print("Iniciando treinamento...")
    print(f"Seed utilizado: {SEED}")
    
    # Uso de 5 épocas
    # Batch size fixo e shuffle=True mas a seed garante reprodutibilidade
    history = model.fit(
        x_train, y_train, 
        batch_size=128, 
        epochs=5, 
        validation_split=0.1,
        shuffle=True,
        verbose=1
    )

    print("\nAvaliando no conjunto de teste...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    
    # Faz predições para calcular métricas adicionais
    y_pred_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calcula métricas usando TensorFlow
    # Converte para one-hot para cálculos
    y_true_one_hot = tf.keras.utils.to_categorical(y_test, 10)
    y_pred_one_hot = tf.keras.utils.to_categorical(y_pred, 10)
    
    # Calcula precisão, recall e F1-score
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    
    precision.update_state(y_true_one_hot, y_pred_one_hot)
    recall.update_state(y_true_one_hot, y_pred_one_hot)
    
    precision_val = precision.result().numpy()
    recall_val = recall.result().numpy()
    f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-7)
    
    print(f"\nResultados no conjunto de teste:")
    print(f"  Acurácia:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precisão:  {precision_val:.4f}")
    print(f"  Recall:    {recall_val:.4f}")
    print(f"  F1-Score:  {f1_val:.4f}")

    # Salva o modelo no formato Keras (.h5)
    model.save("model.h5")
    print("\nModelo salvo como 'model.h5'")
    print(f"\nResultados reproduzíveis garantidos com SEED={SEED}")

if __name__ == "__main__":
    train_model()