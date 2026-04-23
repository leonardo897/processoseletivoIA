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
    
    all_predictions = []
    total_predictions = len(x_test)
    
    # Seed para reprodutibilidade na ordem de avaliação
    np.random.seed(42)
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
        all_predictions.append(predicted_label)
        
        # Progresso
        if (idx + 1) % 1000 == 0:
            print(f"  Avaliados {idx + 1}/{total_predictions} exemplos...")
    
    all_predictions = np.array(all_predictions)
    
    # Calcula métricas usando TensorFlow
    accuracy = np.mean(all_predictions == y_test)
    
    # Converte para one-hot para métricas
    y_true_one_hot = tf.keras.utils.to_categorical(y_test, 10)
    y_pred_one_hot = tf.keras.utils.to_categorical(all_predictions, 10)
    
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    
    precision_metric.update_state(y_true_one_hot, y_pred_one_hot)
    recall_metric.update_state(y_true_one_hot, y_pred_one_hot)
    
    precision = precision_metric.result().numpy()
    recall = recall_metric.result().numpy()
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def apply_optimization(model, optimization_type, x_test=None):
    """Aplica diferentes tipos de otimização e retorna o modelo convertido"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if optimization_type == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif optimization_type == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    return converter.convert()

def optimize_model():
    # Define seed para reprodutibilidade
    SEED = 42
    set_seeds(SEED)
    
    print("=" * 60)
    print("OTIMIZAÇÃO E COMPARAÇÃO DE MODELOS - MÚLTIPLAS TÉCNICAS")
    print(f"SEED configurada: {SEED}")
    print("=" * 60)
    
    # Carrega dataset para teste
    print("\n1. Carregando dataset MNIST para avaliação...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Pré-processamento
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., tf.newaxis]
    
    print(f"Shape dos dados de teste: {x_test.shape}")
    
    # Carrega modelo original
    print("\n2. Carregando modelo original 'model.h5'...")
    model = tf.keras.models.load_model("model.h5")
    
    # Recompila o modelo para garantir reprodutibilidade
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    
    # Avalia modelo original
    print("\n3. Avaliando modelo original...")
    original_loss, original_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Acurácia do modelo original: {original_accuracy:.4f} ({original_accuracy*100:.2f}%)")
    
    # Testa as diferentes técnicas de otimização
    print("\n4. Testando múltiplas técnicas de otimização...")
    
    optimization_techniques = {
        "dynamic": "Dynamic Range Quantization",
        "float16": "Float16 Quantization"
    }
    
    results = {}
    best_model = None
    best_technique = None
    best_accuracy = 0
    best_metrics = None
    
    for technique, description in optimization_techniques.items():
        print(f"\nTestando: {description}")
        try:
            tflite_model = apply_optimization(model, technique, x_test)
            
            # Salva temporariamente
            temp_file = f"model_{technique}.tflite"
            with open(temp_file, "wb") as f:
                f.write(tflite_model)
            
            # Avalia modelo
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            print(f"Avaliando modelo {technique}...")
            metrics = evaluate_tflite_model(interpreter, x_test, y_test)
            
            # Registra resultados
            size_kb = os.path.getsize(temp_file) / 1024
            results[technique] = {
                'description': description,
                'metrics': metrics,
                'size_kb': size_kb,
                'model_data': tflite_model
            }
            
            print(f"Acurácia: {metrics['accuracy']*100:.2f}%")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"Tamanho:  {size_kb:.2f} KB")
            
            # Seleciona o melhor baseado em acurácia
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_technique = technique
                best_metrics = metrics
                best_model = tflite_model
                
        except Exception as e:
            print(f"Erro na técnica {technique}: {str(e)}")
            continue
    
    # Salva apenas o melhor modelo como model.tflite
    print("\n5. Selecionando a melhor técnica de otimização...")
    if best_model is not None:
        with open("model.tflite", "wb") as f:
            f.write(best_model)
        print(f"Melhor técnica: {optimization_techniques[best_technique]}")
        print(f"Modelo salvo como 'model.tflite'")
    else:
        print("Nenhuma técnica de otimização funcionou!")
        return
    
    # Limpa arquivos temporários
    for technique in optimization_techniques.keys():
        temp_file = f"model_{technique}.tflite"
        if os.path.exists(temp_file) and technique != best_technique:
            os.remove(temp_file)
    
    # Comparação de tamanho
    print("\n6. Comparação de tamanho dos arquivos:")
    size_h5 = os.path.getsize("model.h5") / 1024  # KB
    size_tflite = os.path.getsize("model.tflite") / 1024  # KB
    reduction = (1 - (size_tflite / size_h5)) * 100
    
    print(f"Modelo original (.h5):  {size_h5:.2f} KB")
    print(f"Modelo otimizado (.tflite): {size_tflite:.2f} KB")
    print(f"Redução de tamanho: {reduction:.1f}%")
    
    # Comparação final
    print("\n" + "=" * 60)
    print("RESUMO DA COMPARAÇÃO - MELHOR TÉCNICA SELECIONADA")
    print("=" * 60)
    print(f"SEED utilizada:      {SEED}")
    print(f"Melhor técnica:      {optimization_techniques[best_technique]}")
    print(f"\nMÉTRICAS DO MODELO ORIGINAL:")
    print(f"   Acurácia:  {original_accuracy*100:.2f}%")
    print(f"\nMÉTRICAS DO MODELO OTIMIZADO:")
    print(f"   Acurácia:  {best_metrics['accuracy']*100:.2f}%")
    print(f"   Precisão:  {best_metrics['precision']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    print(f"   F1-Score:  {best_metrics['f1_score']:.4f}")
    print(f"\nDIFERENÇAS:")
    accuracy_loss = (original_accuracy - best_metrics['accuracy']) * 100
    print(f"   Perda de acurácia:   {accuracy_loss:.3f} pontos percentuais")
    print(f"\nCOMPARAÇÃO DE TAMANHO:")
    print(f"   Tamanho Original:    {size_h5:.2f} KB")
    print(f"   Tamanho TFLite:      {size_tflite:.2f} KB")
    print(f"   Compressão:          {reduction:.1f}%")
    
    # Tabela comparativa de todas as técnicas
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DE TODAS AS TÉCNICAS TESTADAS")
    print("=" * 60)
    print(f"{'Técnica':<20} {'Acurácia':<12} {'F1-Score':<12} {'Tamanho (KB)':<15} {'Status':<10}")
    print("-" * 60)
    for technique, data in results.items():
        status = "MELHOR" if technique == best_technique else "OK"
        print(f"{data['description']:<20} "
              f"{data['metrics']['accuracy']*100:>6.2f}%     "
              f"{data['metrics']['f1_score']:>6.4f}     "
              f"{data['size_kb']:>10.2f}      "
              f"{status:<10}")
    
    print("\nOs resultados são REPRODUTÍVEIS - Cada execução produzirá os mesmos valores!")

if __name__ == "__main__":
    optimize_model()