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
    
    # Para o TensorFlow garantir operações determinísticas
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
        metrics=["accuracy"]
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
    print(f"\nAcurácia final no teste: {test_acc:.4f}")

    # Salva o modelo no formato Keras (.h5)
    model.save("model.h5")
    print("Modelo salvo como 'model.h5'")
    print(f"\nResultados reproduzíveis garantidos com SEED={SEED}")

if __name__ == "__main__":
    train_model()