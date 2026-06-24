# Contextualização — Entregas 11 e 12

## Entrega 11 — CNN LeNet no MNIST

### Por que CNN em vez de rede densa?
Redes densas ignoram a relação espacial entre pixels: convertem a imagem 2D em vetor 1D antes de processá-la. Uma CNN mantém a estrutura espacial aplicando filtros locais que deslizam pela imagem, detectando padrões como bordas e curvas independente de onde aparecem.

### Arquitetura usada (apostila cnn3.py)
Dois blocos conv+pooling seguidos de camadas densas:

```
Conv2D(20, 5×5) → MaxPool(2×2) → Conv2D(40, 5×5) → MaxPool(2×2)
→ Flatten(640) → Dense(200) → Dense(10, softmax)
```

- Primeira conv: 20 filtros 5×5 aplicados à imagem 28×28, produzindo 20 mapas 24×24
- MaxPool 2×2: reduz para 12×12
- Segunda conv: 40 filtros 5×5 operando nos 20 mapas anteriores simultaneamente
- MaxPool 2×2: reduz para 4×4×40 = 640 atributos
- Camadas densas: classificador final com 10 saídas (dígitos 0–9)

### O que o script faz (`conv_mnist.py`)
1. **Carrega `cnn3.keras`** (modelo pré-treinado disponibilizado pelo prof.) e avalia no conjunto de teste MNIST — resultado esperado ≈ 99,5% de acurácia
2. **Visualiza os 20 filtros 5×5** da primeira camada (`resultados/filtros_conv1.png`) — alguns detectam bordas verticais, horizontais, diagonais
3. **Mapas de ativação** (`resultados/ativacoes_conv1.png`): usa duas imagens do zip (`at_1_002_dig.png`, `at_3_018_dig.png`) e mostra como cada filtro responde ao dígito "1" e ao "3" — filtros de borda vertical ativam no "1", filtros de curva ativam no "3"
4. **Treina rede do zero** com a mesma arquitetura (30 épocas, Adam, batch 100) e compara com o modelo pronto
5. **Curva de treinamento** (`resultados/curva_treinamento.png`) e **matriz de confusão** (`resultados/matriz_confusao.png`) — erros típicos entre pares como 4/9 e 3/5

---

## Entrega 12 — CIFAR-10 com pré-treinamento

### Por que CIFAR-10 é mais difícil que MNIST?
Imagens coloridas 32×32 de 10 classes reais (avião, automóvel, gato, etc.). Um ser humano acerta ~94% — a variação intra-classe é enorme (um gato pode estar deitado, em pé, de frente, de costas).

### Três abordagens comparadas

**Baseline LeNet** (`resultados/curva_baseline.png`)
Mesma estrutura da entrega 11 adaptada para imagens coloridas. Acurácia típica: 72–74%. Sofre overfitting claro — treino chega a 95%+ mas validação estagna.

**EfficientNetB0 com fine-tuning** (`resultados/curva_efficientnet.png`)
Rede treinada no ImageNet (1,2 milhão de imagens, 1000 classes) usada como extrator de atributos para CIFAR-10. As imagens são redimensionadas de 32×32 para 96×96 como camada do modelo.

Processo em duas fases:
- Fase 1 (5 épocas): base congelada, treina só a cabeça nova → aprende a mapear atributos do ImageNet para as 10 classes do CIFAR-10
- Fase 2 (10 épocas): descongela as últimas 30 camadas da base com learning rate menor (1e-4) → adapta os filtros ao domínio do CIFAR-10

Resultado esperado: 85–90%+ de acurácia. O ganho vem dos filtros que a rede já aprendeu no ImageNet (bordas, texturas, formas) sendo reutilizados.

**Baseline + data augmentation** (`resultados/curva_baseline_da.png`)
Mesma LeNet baseline, mas com flip horizontal, rotação ±10% e zoom ±10% aplicados durante o treino. Reduz overfitting e ganha alguns pontos percentuais sobre o baseline puro sem aumentar a capacidade da rede.

### Comparação final (`resultados/comparacao_final.png`)
Gráfico de barras com as três acurácias lado a lado. O ponto principal: transfer learning com EfficientNetB0 supera amplamente treinar do zero, mesmo com dados limitados, porque reutiliza representações já aprendidas.

---

## Conceitos-chave para citar

| Conceito | Onde aparece |
|---|---|
| Filtros convolucionais como detectores de borda | Visualização dos filtros (entrega 11) |
| Transfer learning | EfficientNetB0 fine-tuning (entrega 12) |
| Overfitting | Curvas de treino vs. validação (ambas) |
| Data augmentation como regularização | Baseline DA vs. baseline puro |
| Feature maps | Mapas de ativação do dígito 1 e 3 |
