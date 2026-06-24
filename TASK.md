# Entrega 11 e 12 — CNN/LeNet (MNIST) + CIFAR-10 com Modelos Pré-treinados

## Contexto

Disciplina PSI3471 — Processamento de Imagens, Prof. Hae Yong Kim (USP).
Branch: `entrega/11-12` | Worktree: `worktrees/entrega-11-12/`

A estrutura espelha o padrão das entregas anteriores (9_hae, 10_hae).
**Nunca use `pip` diretamente — sempre `uv pip install` ou `uv add`.**

## Apostilas de referência

- `assets/11/convkeras-ead.pdf` — Rede neural convolucional, LeNet, MNIST
- `assets/12/cifar-ead.pdf` — CIFAR-10, data augmentation, modelos pré-treinados

## Arquivos disponíveis

| Pasta | Arquivo | Descrição |
|---|---|---|
| `assets/11/` | `convkeras-ead.pdf` | Apostila da aula 11 |
| `assets/11/` | `cnn3.keras` | Rede treinada LeNet-like (MNIST) |
| `assets/11/` | `convkeras.zip` | Imagens e exemplos da aula 11 |
| `assets/12/` | `cifar-ead.pdf` | Apostila aulas 11 pt2 / 12 |
| `assets/12/` | `cifar_pretreinado.zip` | Dataset CIFAR-10 pré-processado |
| `assets/12/` | `cifar-exemplos.png` | Grade de exemplos CIFAR-10 |
| `assets/12/` | `cifar-caogato.png` | Exemplos cachorro vs. gato |

O `cekeikon` do prof. Hae está disponível via PyPI. Use `assets/utils/cekeikon_opencv.pdf`
como referência. Instale com `uv pip install cekeikon`.

## Objetivos por entrega

### Entrega 11 — `entregas/11_hae/`

Implementar uma CNN no estilo LeNet para classificação do MNIST:

1. Carregar o modelo treinado `cnn3.keras` e avaliar no conjunto de teste
2. Visualizar os filtros da primeira camada convolucional
3. Visualizar os mapas de ativação (feature maps) para imagens de teste
4. Treinar uma rede do zero com a arquitetura da apostila e comparar com o modelo pronto
5. Plotar a curva de treinamento (accuracy e loss) e a matriz de confusão

Arquivo de saída principal: `entregas/11_hae/conv_mnist.py` (script executável via `uv run`)

### Entrega 12 — `entregas/12_hae/`

CIFAR-10 com modelos pré-treinados (ImageNet) + data augmentation:

1. Explorar o dataset CIFAR-10: distribuição de classes, exemplos
2. Treinar um modelo simples (baseline) sem pré-treinamento
3. Aplicar fine-tuning de um modelo pré-treinado no ImageNet (MobileNetV2 ou EfficientNetB0
   — escolha a arquitetura mais eficiente)
4. Comparar os resultados: baseline vs. pré-treinado
5. Aplicar data augmentation (flip, rotação, zoom) e re-treinar; medir o ganho

Arquivo de saída principal: `entregas/12_hae/cifar_pretrained.py`

## Padrão de projeto

```
entregas/
  11_hae/
    pyproject.toml        # nome: "11-hae-conv", deps: cekeikon, keras, etc.
    conv_mnist.py         # script principal
    resultados/           # imagens geradas, métricas
  12_hae/
    pyproject.toml        # nome: "12-hae-cifar"
    cifar_pretrained.py
    resultados/
```

Inicialize os ambientes com `uv init` + `uv add cekeikon tensorflow` dentro de cada pasta.
Use `uv run python script.py` para executar. **Nunca `pip install`.**

## Uso dos assets do projeto

- Reuse `assets/utils/cekeikon_opencv.pdf` como referência da biblioteca
- O padrão de visualização do cekeikon (`cek.mostra()`, `cek.espera()`) deve ser
  preferido em vez de `plt.show()` onde aplicável
- Salve todas as imagens de resultado em `resultados/` com nomes descritivos

## Critérios de qualidade

- Código limpo, sem comentários óbvios — o nome das funções deve ser auto-explicativo
- Resultados reproduzíveis: fixe seeds (`tf.random.set_seed(42)`)
- Gráficos com títulos, eixos e legendas em português
- Relatório de métricas em texto no stdout ao final de cada script

---

## Roteiro de vídeo curto

**Ao finalizar a implementação**, gere um arquivo `entregas/ROTEIRO_11_12.md` com um roteiro
de vídeo de **3 a 5 minutos** explicando o código. O roteiro deve:

1. **Abertura (30s)**: apresentar o problema (classificação de imagens com CNN) e as duas entregas
2. **Entrega 11 — LeNet/MNIST (90s)**:
   - Mostrar a arquitetura da rede no terminal/IDE
   - Executar `conv_mnist.py` ao vivo e comentar a saída
   - Exibir os filtros e feature maps gerados
3. **Entrega 12 — CIFAR-10 (90s)**:
   - Mostrar a diferença entre baseline e fine-tuning
   - Executar `cifar_pretrained.py` e comentar as métricas
   - Exibir a curva de accuracy e a melhora com data augmentation
4. **Fechamento (30s)**: resumo dos resultados, o que funcionou melhor e por quê

O roteiro deve ser escrito em **linguagem coloquial e direta**, como se fosse apresentado
para colegas de curso. Indique os timestamps aproximados para cada seção.
