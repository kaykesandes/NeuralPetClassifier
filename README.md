[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.0+-red.svg)](https://keras.io)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0+-yellow.svg)](https://matplotlib.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.18+-purple.svg)](https://numpy.org)
[![Pillow](https://img.shields.io/badge/Pillow-7.0+-green.svg)](https://python-pillow.org)

# Classificação de Imagens de Gatos e Cachorros com Transfer Learning em Python

> **Resumo:** Classifique automaticamente imagens de gatos e cachorros usando Transfer Learning com Keras/TensorFlow, explorando conceitos de CNN, métricas e visualização interativa.

## 🗂️ Índice
- [Descrição Geral](#descrição-geral)
- [Recursos](#recursos)
- [Fluxo de Dados](#fluxo-de-dados)
- [Arquitetura da Rede Neural](#arquitetura-da-rede-neural)
- [Transfer Learning](#transfer-learning)
- [Treinamento e Loss](#treinamento-e-loss)
- [Métrica de Acurácia](#métrica-de-acurácia)
- [Visualização dos Resultados](#visualização-dos-resultados)
- [Tratamento dos Dados](#tratamento-dos-dados)
- [Instalação](#instalação)
- [Uso](#uso)
- [Sugestões de Melhoria](#sugestões-de-melhoria)
- [Referências Matemáticas e Técnicas](#referências-matemáticas-e-técnicas)
- [Autor](#autor)

---

## Descrição Geral

Este projeto utiliza redes neurais convolucionais (CNNs) e Transfer Learning para classificar imagens de gatos e cachorros. O pipeline cobre desde o tratamento dos dados até a avaliação dos resultados, com explicações matemáticas, arquitetura, fluxo de dados e dicas avançadas.

---

## 1. Neurônio Artificial

Um neurônio artificial recebe entradas $(x_1, x_2, ..., x_n)$, multiplica cada uma por um peso $(w_1, w_2, ..., w_n)$, soma tudo e aplica uma função de ativação:

$$
\text{saída} = f(w_1 x_1 + w_2 x_2 + ... + w_n x_n + b)
$$

Onde:

- $f$ é a função de ativação (ex: ReLU, Sigmoid)
- $b$ é o bias

**Exemplo:**
Se $x = [0.5, 0.2]$, $w = [0.8, -0.3]$, $b = 0.1$, $f(x) = \text{ReLU}(x)$:

$$
\text{saída} = \text{ReLU}(0.8 \times 0.5 + (-0.3) \times 0.2 + 0.1) = \text{ReLU}(0.4 - 0.06 + 0.1) = \text{ReLU}(0.44) = 0.44
$$

---

## 2. Fluxo de Dados

- **Entrada:** Imagens são lidas das pastas `PetImages/Cat` e `PetImages/Dog`.
- **Pré-processamento:** Redimensionamento para 224x224, normalização dos pixels para [0, 1], remoção de arquivos corrompidos.
- **Divisão:** Os dados são embaralhados e divididos em treino, validação e teste.
- **Geradores:** O `ImageDataGenerator` carrega os dados em lotes, evitando sobrecarga de memória.
- **Treinamento:** O modelo aprende a partir dos dados de treino e validação.
- **Avaliação:** O desempenho é medido no conjunto de teste.
- **Visualização:** Exemplos de predição são exibidos, mostrando imagem, rótulo verdadeiro e predição.

---

## 3. Arquitetura da Rede Neural

- **Camadas Convolucionais (Conv2D):** Extraem padrões locais (bordas, texturas).
- **Pooling (MaxPooling2D):** Reduz a dimensionalidade, mantendo informações relevantes.
- **Dropout:** Evita overfitting desligando neurônios aleatoriamente.
- **Flatten:** Transforma matriz em vetor.
- **Dense:** Realiza a classificação final.
- **Softmax:** Gera probabilidades para cada classe.

**Exemplo de saída Softmax para 2 classes:**

$$
\text{softmax}(z_1, z_2) = \left[ \frac{e^{z_1}}{e^{z_1} + e^{z_2}}, \frac{e^{z_2}}{e^{z_1} + e^{z_2}} \right]
$$

Se $z_1 = 2$, $z_2 = 1$:

$$
\begin{align*}
e^{2} &\approx 7.39 \\
e^{1} &\approx 2.72 \\
\text{softmax} &= \left[ \frac{7.39}{7.39+2.72}, \frac{2.72}{7.39+2.72} \right] \approx [0.73, 0.27]
\end{align*}
$$

---

## 4. Transfer Learning

Usa-se um modelo pré-treinado (ex: VGG16) que já aprendeu a extrair características de imagens. Adiciona-se uma nova camada densa para classificar gatos e cachorros. As camadas do modelo base ficam congeladas:

```python
for layer in base_model.layers:
    layer.trainable = False
```

Apenas a nova camada é treinada, aproveitando o conhecimento do modelo base.

---

## 5. Treinamento e Loss

- **Forward Pass:** Imagem passa pela rede, gerando predição.
- **Loss Function:** Mede o erro entre predição e rótulo verdadeiro:

$$
\text{categorical\_crossentropy} = -\sum y_{true} \cdot \log(y_{pred})
$$

**Exemplo:**
Se $y_{true} = [1, 0]$, $y_{pred} = [0.8, 0.2]$:

$$
\text{loss} = -[1 \cdot \log(0.8) + 0 \cdot \log(0.2)] = -\log(0.8) \approx 0.22
$$

- **Backward Pass:** Ajusta os pesos para minimizar o erro.
- **Epochs:** Repete o processo várias vezes.

---

## 6. Métrica de Acurácia

A acurácia mede a proporção de predições corretas:

$$
\text{accuracy} = \frac{n_{acertos}}{n_{total}}
$$

**Exemplo:**
Se o modelo acertou 85 de 100 imagens:

$$
\text{accuracy} = \frac{85}{100} = 0.85 \text{ (ou 85%)}
$$

---

## 7. Visualização dos Resultados

O código exibe exemplos do teste, mostrando:

- Imagem
- Rótulo verdadeiro
- Predição do modelo

Permite navegação interativa para visualizar grupos de imagens.

---

## 8. Tratamento dos Dados

- Remoção automática de imagens corrompidas.
- Limitação do número de imagens por categoria.
- Normalização dos dados.

---

## 9. Instalação

1. Ative o ambiente virtual:
   ```powershell
   .\.venv\Scripts\activate
   ```
2. Instale as dependências:
   ```powershell
   pip install tensorflow keras matplotlib numpy pillow
   ```

---

## 10. Uso

Execute o script principal:
   ```powershell
   python main.py
   ```

---

## 11. Sugestões de Melhoria

- Adicionar modelos pré-treinados diferentes.
- Implementar augmentação de dados.
- Salvar o modelo treinado.
- Adicionar interface web para visualização.
- Testar diferentes arquiteturas e hiperparâmetros.

---

## 12. Referências Matemáticas e Técnicas

- [Softmax Function](https://en.wikipedia.org/wiki/Softmax_function)
- [Cross-Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy)
- [Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning)
- [CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network)

---

## Autor

**Kayke s.**
- 📧 Email: kaykegy@proton.me   
- 💼 LinkedIn: <a href="https://www.linkedin.com/in/kayke-g-171b7223a/">Linkedin-Kayke S.</a>
- 🐙 GitHub: <a href="https://github.com/kaykesandes">GitHub-Kayke S.</a>

---

Agradeço por visitar este projeto! Sinta-se à vontade para contribuir, sugerir melhorias ou adaptar para seus próprios estudos. Bons experimentos e sucesso na sua jornada com Deep Learning e Visão Computacional!
