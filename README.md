# Processo Seletivo – Intensivo Maker | AI

Bem-vindo(a) à **etapa prática do processo seletivo para o Intensivo Maker**.

Esta atividade tem como objetivo avaliar competências técnicas relacionadas a **Machine Learning**, **Visão Computacional** e **Otimização de modelos para sistemas embarcados (Edge AI)**, a partir da aplicação prática dos conhecimentos adquiridos nos cursos EAD da etapa anterior.

> 🎯 **Importante**  
> O foco deste desafio é avaliar sua capacidade de **projetar, treinar e otimizar um modelo de IA**.  

---

## 📌 Navegação Rápida

- 🏁 [Passo 0 – Antes de Tudo](#-passo-0-antes-de-tudo)
- ⚙ [Passo 1 – Preparando o Ambiente](#-passo-1-preparando-o-ambiente)
- 💻 [Passo 2 – O Desafio Técnico](#-passo-2-o-desafio-técnico)
  - 🎯 [Conjunto de Dados](#-conjunto-de-dados)
  - 📂 [Estrutura do Projeto](#-estrutura-do-projeto)
  - 📚 [Material de Apoio](#-material-de-apoio)
  - ⚖️ [Critérios de Avaliação](#️-critérios-de-avaliação)
- 📤 [Passo 3 – Instruções de Entrega](#-passo-3-instruções-de-entrega)
  - 📝 [Relatório do Candidato](#-relatório-do-candidato)

---

## 🏁 Passo 0: Antes de Tudo

Caso você **nunca tenha utilizado Git ou GitHub**, não se preocupe.  
Siga atentamente as etapas abaixo.


### 1️⃣ Criação de Conta no GitHub

1. Acesse: https://github.com  
2. Clique em **Sign up**  
3. Crie sua conta gratuita seguindo as instruções da plataforma  

(*O GitHub será utilizado para envio, versionamento e correção automática do seu projeto.*)


### 2️⃣ Instalação do Git

O **Git** é a ferramenta que permite versionar e enviar seu código para o GitHub.

- **Windows**  
  Baixe e instale o **Git Bash**:  
  https://git-scm.com/downloads

- **Linux / macOS**  
  Verifique se o Git já está instalado:
  ```bash
  git --version
  ```

---

## ⚙ Passo 1: Preparando o Ambiente

Para desenvolver o desafio, você deverá criar uma cópia deste repositório.

### 1️⃣ Fork do Repositório

<img width="219" height="45" alt="image" src="https://github.com/user-attachments/assets/5d629626-513a-445c-ba0f-e5bb3e225187" />

1. No canto superior direito desta página, clique em **Fork**  
2. Uma cópia deste repositório será criada no **seu perfil do GitHub**
(*O Fork permite que você trabalhe de forma independente sem alterar o repositório original.*)



### 2️⃣ Clone do Repositório

<img width="149" height="52" alt="image" src="https://github.com/user-attachments/assets/abbd331b-a005-4633-89c6-afd16acbe828" />

No repositório do **seu Fork**, clique em **<> Code**, copie a URL e execute:

```bash
git clone https://github.com/SEU_USUARIO/nome-do-repositorio.git
cd nome-do-repositorio
```
(*O comando `git clone` cria uma cópia do repositório.*)



### 3️⃣ Preparação do Ambiente de Execução

Você pode executar o projeto de **Três formas**. Escolha apenas uma.



#### Opção A – Ambiente Python Local 
Requisitos:
- Python **3.10 ou 3.11**
- pip

Instale as dependências com:

```bash
pip install -r requirements.txt
```



#### Opção B – Dev Container 
Este repositório inclui um **Dev Container** para facilitar a criação de um ambiente Python padronizado.

**Requisitos**
- VS Code
- Docker instalado
- Extensão **Dev Containers**

**Passos**
1. Abra o repositório no VS Code  
2. Selecione **“Reopen in Container”**  
3. Aguarde a criação automática do ambiente  

➡️ As dependências serão instaladas automaticamente.


#### Opção C - via browser
Você também pode abrir o container via github codespace

1. Clique em **<> Code**
2. Clique em **Codespaces**
3. Clique em **Create codespace on image**

<img width="482" height="436" alt="image" src="https://github.com/user-attachments/assets/37a1e99d-66d2-4730-b824-26f834bd8cc3" />


>  Será aberto uma instância do VS Code no seu navegador com o container configurado


---

## 💻 Passo 2: O Desafio Técnico

O desafio consiste em desenvolver um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos**, e posteriormente **otimizá-lo para execução em dispositivos Edge**, como sistemas embarcados e IoT.

O foco não é apenas obter alta acurácia, mas também **compreender o fluxo completo**:

**treinamento → salvamento → conversão → otimização**



### 🎯 Conjunto de Dados

Será utilizado o dataset **MNIST**, composto por imagens de dígitos manuscritos de **0 a 9**.
<img width="500" height="294" alt="image" src="https://github.com/user-attachments/assets/f323b4cc-d759-4e05-bb58-13e4d6dc7e5b" />

✔️ O dataset já está disponível na biblioteca **TensorFlow/Keras**, não sendo necessário download manual.

📌 *O MNIST é amplamente utilizado para introdução à Visão Computacional e Redes Neurais.*



###  ✅ Requisitos Obrigatórios

**Etapa 1:**  Treinamento do Modelo (`train_model.py`)

Implemente no arquivo `train_model.py` um código que realize:

- Carregamento do dataset MNIST via TensorFlow
- Construção e treinamento de um modelo de classificação baseado em **Redes Neurais Convolucionais (CNN)**  
  (utilizando camadas `Conv2D` e `MaxPooling`)
- Treinamento do modelo
- Exibição da **acurácia final** no terminal
- Salvamento do modelo treinado no formato **Keras** (`.h5`)

(*O modelo salvo será utilizado na etapa de otimização.*)



**Etapa 2:** Otimização do Modelo (`optimize_model.py`)

No arquivo `optimize_model.py`, implemente:

- Carregamento do modelo treinado
- Conversão para **TensorFlow Lite (`.tflite`)**
- Aplicação de técnica de otimização, como:
  - **Dynamic Range Quantization**

(**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de **Edge AI**.)



### 📂 Estrutura do Projeto

⚠️ **Atenção:**  
A estrutura e os nomes dos arquivos **não devem ser alterados**.

```plaintext
seu-repositorio/
├── .github/
│   └── workflows/
│       └── ci.yml            # 🤖 Pipeline de correção automática (NÃO ALTERAR)
├── .devcontainer/            # 🐳 Dev Container (opcional)
│   └── devcontainer.json
├── train_model.py            # ✏️ Treinamento do modelo
├── optimize_model.py         # ✏️ Conversão e otimização
├── requirements.txt          # 📄 Dependências do projeto
├── model.h5                  # 🤖 Modelo treinado (gerado)
├── model.tflite              # ⚡ Modelo otimizado (gerado)
└── README.md                 # 📝 Relatório final do candidato
```



### ⚠️ Restrições e Considerações de Engenharia

Este desafio é avaliado automaticamente por meio de um pipeline de
**integração contínua (CI)**, executado em um ambiente controlado e com
restrições de recursos computacionais.

Você **não precisa conhecer GitHub Actions** para realizar o desafio.
No entanto, é importante respeitar as diretrizes abaixo.

**Diretrizes para o Modelo**

- O modelo deve ser uma **CNN simples**, adequada para **Edge AI**
- Evite arquiteturas muito profundas ou complexas
- Recomenda-se utilizar **até 3 camadas convolucionais**
- **Não utilize modelos pré-treinados**
- Número de épocas **limitado** (ex: até 5)

#### Diretrizes de Execução

- Treinamento apenas em **CPU**
- Tempo total reduzido (compatível com CI)
- Código deve executar do início ao fim **sem intervenção manual**

> **Importante:**  
> O objetivo não é obter a maior acurácia possível, mas sim demonstrar
> **engenharia eficiente**, compatível com ambientes automatizados e
> restrições típicas de aplicações reais de Edge AI.



### 📚 Material de Apoio

Os cursos realizados na etapa anterior **devem ser utilizados como referência**.

- 📘 **Fundamentos de Inteligência Artificial para Sistemas Embarcados**
- 👁️ **Sistemas de Visão Computacional Embarcada**
- ⚙️ **Otimização de Modelos em Sistemas Embarcados**

(*Os exemplos apresentados nesses cursos podem ser adaptados e reutilizados neste desafio.*)



### ⚖️ Critérios de Avaliação

A avaliação considerará:

- **Funcionalidade**  
  Execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`

- **Edge AI**  
  Conversão correta para `.tflite` e aplicação de técnica de otimização

- **Documentação**  
  Preenchimento adequado do relatório (README.md)

---

## 📤 Passo 3: Instruções de Entrega

### ✔️ Validação 

Antes do envio, execute os scripts e confirme a geração dos arquivos:
- `model.h5`
- `model.tflite`



### ⬆️ Envio do Código

```bash
git add .
git commit -m "Entrega do desafio técnico - Seu Nome"
git push origin main
```



### 🔍 Verificação Automática

1. Acesse a aba **Actions** no GitHub  
2. Verifique se o workflow foi executado com sucesso (✅)  
3. Em caso de erro (❌), consulte os logs, corrija e envie novamente

<img width="807" height="363" alt="image" src="https://github.com/user-attachments/assets/d991d35b-2bc2-48f7-9ac7-cf5ca9dc452a" />



### 📎 Submissão Final

Copie o link do seu repositório e envie conforme orientações do processo seletivo no Moodle.

---

## 📝 Relatório do Candidato

**👤 Identificação:** 
**Nome Completo:** Leonardo de Oliveira Sales Vieira

### 1️⃣ Resumo da Arquitetura do Modelo

O modelo implementado é uma Rede Neural Convolucional (CNN) simples e eficiente, adequada para aplicações de Edge AI. A arquitetura consiste em:

- **Camada de Input:** Recebe imagens de 28×28 pixels em escala de cinza (1 canal)
- **Primeiro Bloco Convolucional:** 
  - Conv2D com 8 filtros de tamanho 3×3 e ativação ReLU
  - MaxPooling2D com pool size 2×2 (reduz dimensionalidade pela metade)
- **Segundo Bloco Convolucional:**
  - Conv2D com 16 filtros de tamanho 3×3 e ativação ReLU
  - MaxPooling2D com pool size 2×2
- **Camada de Flatten:** Converte o tensor 2D em vetor 1D (400 neurônios)
- **Camada Densa Intermediária:** 32 neurônios com ativação ReLU
- **Camada de Saída:** 10 neurônios com ativação softmax para classificação dos dígitos de 0 a 9

**Total de parâmetros:** 14.410 (56,29 KB)

**Garantia de Reprodutibilidade:**
Para assegurar resultados consistentes e determinísticos, foram implementadas as seguintes práticas:
- Configuração de seed global (valor: 42) para Python random, NumPy e TensorFlow
- Inicializadores `GlorotUniform` com seed explícita em todas as camadas
- Uso de `tf.keras.utils.set_random_seed()` para operações do Keras
- Ativação de operações determinísticas via `tf.config.experimental.enable_op_determinism()`
- Taxa de aprendizado fixa no otimizador Adam (0.001)

A arquitetura foi projetada intencionalmente com poucas camadas e número reduzido de filtros para manter o modelo leve e adequado para dispositivos embarcados, com o uso de 3 camadas convolucionais.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow 2.x** - Framework principal para construção, treinamento e conversão do modelo
- **NumPy** - Manipulação de arrays e operações numéricas
- **Random** - Configuração de seeds para reprodutibilidade
- **OS** - Operações de sistema de arquivos (built-in)

O TensorFlow foi utilizado tanto para o treinamento da CNN quanto para a conversão e quantização do modelo para o formato TensorFlow Lite.

### 3️⃣ Técnica de Otimização do Modelo

A otimização do modelo foi realizada utilizando **Dynamic Range Quantization** através do TensorFlow Lite Converter. 

**Processo de otimização:**
1. Carregamento do modelo treinado no formato `.h5`
2. Configuração do `TFLiteConverter` com a flag `tf.lite.Optimize.DEFAULT`
3. Conversão do modelo para o formato `.tflite` com quantização aplicada

**Garantia de Reprodutibilidade na Otimização:**
- Mesma seed (42) utilizada no treinamento é aplicada antes da conversão
- Recompilação do modelo com configurações idênticas para avaliação
- Seed explícita antes da inferência com TFLite para ordem determinística de avaliação
- Uso do conversor experimental estável (`experimental_new_converter=True`)

A quantização dinâmica reduz os pesos do modelo de float32 para int8 durante a inferência, diminuindo significativamente o tamanho do arquivo e o uso de memória, mantendo a maior parte da precisão original. Esta técnica é particularmente eficaz para Edge AI pois:
- Reduz o footprint de memória
- Acelera a inferência em CPUs com suporte a operações int8
- Não requer dados de calibração (diferente da quantização estática)

### 4️⃣ Resultados Obtidos

**Treinamento:**
- **Seed utilizada:** 42
- Acurácia no conjunto de teste: **98,11%**
- Épocas utilizadas: 5
- Tempo médio por época: ~3-5 segundos
- Reprodutibilidade garantida: múltiplas execuções produzem o mesmo resultado

**Comparação entre Modelos:**

| Métrica | Modelo Original (.h5) | Modelo Otimizado (.tflite) |
|---------|----------------------|---------------------------|
| Acurácia | 98,11% | 98,11% |
| Tamanho do Arquivo | 208,17 KB | 20,35 KB |
| Redução de Tamanho | - | **90,2%** |
| Perda de Precisão | - | **0,0%** |
| Seed | 42 | 42 |

**Análise dos Resultados:**
- O modelo atingiu alta acurácia (98,11%) com arquitetura simples e apenas 5 épocas
- A implementação de seeds garante que estes resultados sejam **100% reprodutíveis**
- A otimização para TensorFlow Lite resultou em uma redução de **90,2%** no tamanho do arquivo
- A perda de precisão foi **insignificante**
- O modelo final tem apenas **20,35 KB**, ideal para sistemas embarcados com recursos limitados

### 5️⃣ Comentários Adicionais (Opcional)

**Decisões Técnicas Importantes:**
- A escolha de apenas 8 e 16 filtros nas camadas convolucionais (em vez de 32 ou 64 típicos) foi deliberada para manter o modelo leve
- O uso de MaxPooling após cada convolução reduz progressivamente a dimensionalidade, diminuindo o número de parâmetros na camada densa
- A normalização dos pixels para o intervalo [0, 1] foi essencial para estabilizar o treinamento
- **Implementação de seeds:** Para garantir reprodutibilidade precisa e facilitar validação dos resultados
- Inicializadores com seed explícita em cada camada garantem que os pesos iniciais sejam sempre os mesmos

**Dificuldades Encontradas:**
- A configuração inicial do ambiente com TensorFlow no Windows apresentou problemas de compatibilidade
- Foi necessário mudar o Python para versão 3.11 para compatibilidade com TensorFlow
- Garantir a reprodutibilidade exigiu configuração cuidadosa de múltiplas seeds (Python, NumPy, TensorFlow) e inicializadores

**Limitações do Modelo:**
- Modelo treinado apenas com dígitos centralizados do MNIST (fundo preto, dígito branco)
- Pode apresentar queda de performance em imagens do mundo real com ruído, rotação ou escalas diferentes
- Acurácia limitada em dígitos manuscritos muito diferentes do padrão MNIST

**Aprendizados:**
- A quantização dinâmica do TensorFlow Lite é extremamente eficaz para redução de tamanho com perda mínima de precisão
- Modelos CNN simples podem alcançar excelentes resultados sem necessidade de arquiteturas complexas
- O trade-off entre tamanho do modelo e acurácia é favorável usando as técnicas de otimização apropriadas
- A importância de considerar as restrições de hardware desde a fase de design da arquitetura
- Reprodutibilidade é essencial em projetos profissionais
- A utilização de seeds não afeta a performance do modelo, apenas garante consistência nos resultados

**Impacto da Reprodutibilidade:**
- Facilita debugging e validação de modelos
- Permite comparações justas entre diferentes versões do modelo
- Garante que diferentes pessoas obtenham resultados idênticos

**Conclusão:**
O projeto demonstrou com sucesso o fluxo completo de desenvolvimento para Edge AI: desde o design de uma CNN, o treinamento, até a otimização com TensorFlow Lite. O resultado final é um modelo compacto (20 KB) com precisão idêntica ao original.

## 🆘 Suporte

Em caso de dúvidas:

- Consulte o material dos cursos EAD
- Leia atentamente este README
- Analise os logs das GitHub Actions
- Utilize os canais oficiais para contato com os instrutores

Boa sorte no processo seletivo.
****
