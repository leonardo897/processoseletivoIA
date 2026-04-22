import tensorflow as tf
import numpy as np
import random
import os

# Configuração de seeds para reprodutibilidade
def set_seeds(seed=42):
    """Define todas as seeds necessárias para reprodutibilidade"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

def evaluate_tflite_model(interpreter, x_test, y_test):
    """Avalia a acurácia do modelo TFLite"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct_predictions = 0
    total_predictions = len(x_test)
    
    indices = np.arange(total_predictions)
    
    for idx, i in enumerate(indices):
        # Prepara input
        input_data = x_test[i:i+1].astype(np.float32)
        
        # Faz inferência
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Verifica predição
        predicted_label = np.argmax(output_data)
        true_label = y_test[i]
        
        if predicted_label == true_label:
            correct_predictions += 1
            
        # Progresso
        if (idx + 1) % 1000 == 0:
            print(f"  Avaliados {idx + 1}/{total_predictions} exemplos...")
    
    accuracy = correct_predictions / total_predictions
    return accuracy

def optimize_model():
    # Define seed para reprodutibilidade
    SEED = 42
    set_seeds(SEED)
    
    print("=" * 50)
    print("OTIMIZAÇÃO E COMPARAÇÃO DE MODELOS")
    print(f"SEED configurada: {SEED}")
    print("=" * 50)
    
    # Carrega dataset para teste
    print("\n1. Carregando dataset MNIST para avaliação...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Pré-processamento (o mesmo do treinamento)
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., tf.newaxis]
    
    print(f"   Shape dos dados de teste: {x_test.shape}")
    
    # Carrega modelo original
    print("\n2. Carregando modelo original 'model.h5'...")
    model = tf.keras.models.load_model("model.h5")
    
    # Recompila o modelo para garantir reprodutibilidade
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Avalia modelo original
    print("\n3. Avaliando modelo original...")
    original_loss, original_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Acurácia do modelo original: {original_accuracy:.4f} ({original_accuracy*100:.2f}%)")
    
    # Converte para TFLite com quantização
    print("\n4. Convertendo para TensorFlow Lite com Dynamic Range Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Configurações adicionais para reprodutibilidade na conversão
    converter.experimental_new_converter = True
    
    tflite_model = converter.convert()
    
    # Salva modelo otimizado
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    print("   Modelo otimizado salvo como 'model.tflite'")
    
    # Carrega modelo TFLite para avaliação
    print("\n5. Avaliando modelo otimizado (TFLite)...")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Avalia acurácia do modelo TFLite
    tflite_accuracy = evaluate_tflite_model(interpreter, x_test, y_test)
    print(f"   Acurácia do modelo otimizado: {tflite_accuracy:.4f} ({tflite_accuracy*100:.2f}%)")
    
    # Comparação de tamanho
    print("\n6. Comparação de tamanho dos arquivos:")
    import os
    size_h5 = os.path.getsize("model.h5") / 1024  # KB
    size_tflite = os.path.getsize("model.tflite") / 1024  # KB
    reduction = (1 - (size_tflite / size_h5)) * 100
    
    print(f"   Modelo original (.h5):  {size_h5:.2f} KB")
    print(f"   Modelo otimizado (.tflite): {size_tflite:.2f} KB")
    print(f"   Redução de tamanho: {reduction:.1f}%")
    
    # Comparação final
    print("\n" + "=" * 50)
    print("RESUMO DA COMPARAÇÃO")
    print("=" * 50)
    print(f"SEED utilizada:      {SEED}")
    print(f"Acurácia Original:  {original_accuracy*100:.2f}%")
    print(f"Acurácia TFLite:    {tflite_accuracy*100:.2f}%")
    print(f"Diferença:          {(original_accuracy - tflite_accuracy)*100:.3f} pontos percentuais")
    print(f"Tamanho Original:   {size_h5:.2f} KB")
    print(f"Tamanho TFLite:     {size_tflite:.2f} KB")
    print(f"Compressão:         {reduction:.1f}%")

if __name__ == "__main__":
    optimize_model()