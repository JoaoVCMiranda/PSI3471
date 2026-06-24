#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tensorflow[and-cuda]>=2.16",
#   "matplotlib",
#   "scikit-learn",
#   "pillow",
# ]
# ///
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import io
import zipfile
from time import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import optimizers

tf.random.set_seed(42)
np.random.seed(42)

RAIZ = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(RAIZ, '..', '..', 'assets', '11')
RESULTADOS = os.path.join(RAIZ, 'resultados')

# =============================================================================
# Dados
# =============================================================================
(AX, AY), (QX, QY) = mnist.load_data()
nl, nc = AX.shape[1], AX.shape[2]

# cnn3.keras foi treinado com pixels invertidos (255-x) — mantemos dois tensores:
# QX_inv para avaliar o modelo pré-treinado, QX para treinar do zero
AX_inv = np.expand_dims((255 - AX).astype('float32') / 255.0 - 0.5, axis=3)
QX_inv = np.expand_dims((255 - QX).astype('float32') / 255.0 - 0.5, axis=3)
AX = np.expand_dims(AX.astype('float32') / 255.0 - 0.5, axis=3)
QX = np.expand_dims(QX.astype('float32') / 255.0 - 0.5, axis=3)

nclasses = 10
AY2 = keras.utils.to_categorical(AY, nclasses)
QY2 = keras.utils.to_categorical(QY, nclasses)

print(f'Treino: AX{AX.shape}  AY{AY.shape}')
print(f'Teste:  QX{QX.shape}  QY{QY.shape}')

# =============================================================================
# 1. Modelo pré-treinado
# =============================================================================
print('\n=== 1. Modelo pré-treinado (cnn3.keras) ===')
model = keras.models.load_model(os.path.join(ASSETS, 'cnn3.keras'))
model.summary()

score_pretrained = model.evaluate(QX_inv, QY2, verbose=False)
print(f'Acurácia de teste: {score_pretrained[1]*100:.2f}%')
print(f'Erro de teste:     {(1-score_pretrained[1])*100:.2f}%')

# =============================================================================
# 2. Filtros da primeira camada convolucional
# =============================================================================
print('\n=== 2. Visualizando filtros da primeira camada ===')
filters_raw, _ = model.get_layer(index=0).get_weights()
# shape: (5, 5, 1, 20) -> (5, 5, 20)
filters_raw = np.squeeze(filters_raw)
filtros = np.stack([filters_raw[:, :, i] for i in range(filters_raw.shape[2])])  # (20, 5, 5)

fig = plt.figure(figsize=(10, 4))
for i in range(20):
    ax = fig.add_subplot(4, 5, i + 1)
    ax.imshow(filtros[i], vmin=-0.25, vmax=0.25, cmap='gray')
    ax.axis('off')
plt.subplots_adjust(bottom=0.05, right=0.85, top=0.95, hspace=0.1, wspace=0.1)
cax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(plt.cm.ScalarMappable(
    norm=plt.Normalize(-0.25, 0.25), cmap='gray'), cax=cax)
plt.suptitle('Filtros da 1ª Camada Convolucional (5×5)', fontsize=11)
plt.savefig(os.path.join(RESULTADOS, 'filtros_conv1.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Filtros salvos: resultados/filtros_conv1.png')

# =============================================================================
# 3. Mapas de ativação
# =============================================================================
print('\n=== 3. Mapas de ativação ===')
inp = keras.Input(shape=(nl, nc, 1))
intermediate = Model(inputs=inp, outputs=model.get_layer(index=0)(inp))

zip_path = os.path.join(ASSETS, 'convkeras.zip')
nomes_arquivo = ['at_1_002_dig.png', 'at_3_018_dig.png']
rotulos_img   = ['Dígito 1', 'Dígito 3']

imagens = []
with zipfile.ZipFile(zip_path) as zf:
    for nome in nomes_arquivo:
        with zf.open(nome) as f:
            img = Image.open(io.BytesIO(f.read())).convert('L').resize((28, 28))
            arr = (255 - np.array(img, dtype='float32')) / 255.0 - 0.5
            imagens.append(arr)

fig, axes = plt.subplots(2, 21, figsize=(21, 3))
for row, (arr, rotulo) in enumerate(zip(imagens, rotulos_img)):
    x = arr[np.newaxis, :, :, np.newaxis]
    ativacoes = intermediate.predict(x, verbose=0)  # (1, 24, 24, 20)

    axes[row, 0].imshow(arr, cmap='gray', vmin=-0.5, vmax=0.5)
    axes[row, 0].set_title(rotulo, fontsize=7)
    axes[row, 0].axis('off')

    for i in range(20):
        axes[row, i + 1].imshow(ativacoes[0, :, :, i], cmap='gray')
        axes[row, i + 1].axis('off')

plt.suptitle('Mapas de Ativação — 1ª Camada Convolucional', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS, 'ativacoes_conv1.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Mapas de ativação salvos: resultados/ativacoes_conv1.png')

# =============================================================================
# 4. Treinar rede do zero (mesma arquitetura do apostila cnn3.py)
# =============================================================================
print('\n=== 4. Treinando CNN do zero ===')
model_novo = Sequential([
    Conv2D(20, kernel_size=(5, 5), activation='relu', input_shape=(nl, nc, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(40, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(200, activation='relu'),
    Dense(nclasses, activation='softmax'),
])
model_novo.summary()

model_novo.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

t0 = time()
history = model_novo.fit(
    AX, AY2,
    batch_size=100,
    epochs=30,
    verbose=2,
    validation_split=0.1,
)
t1 = time()
print(f'Tempo de treino: {t1-t0:.2f}s')

score_novo = model_novo.evaluate(QX, QY2, verbose=False)
print(f'Acurácia de teste: {score_novo[1]*100:.2f}%')
print(f'Erro de teste:     {(1-score_novo[1])*100:.2f}%')

# =============================================================================
# 5. Curva de treinamento e matriz de confusão
# =============================================================================
print('\n=== 5. Curvas e matriz de confusão ===')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'],     label='treino')
ax1.plot(history.history['val_accuracy'], label='validação')
ax1.set_title('Acurácia por Época — CNN MNIST')
ax1.set_xlabel('Época')
ax1.set_ylabel('Acurácia')
ax1.legend(loc='lower right')

ax2.plot(history.history['loss'],     label='treino')
ax2.plot(history.history['val_loss'], label='validação')
ax2.set_title('Perda por Época — CNN MNIST')
ax2.set_xlabel('Época')
ax2.set_ylabel('Perda')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS, 'curva_treinamento.png'), dpi=150)
plt.close()
print('Curva de treinamento salva: resultados/curva_treinamento.png')

QP2 = model_novo.predict(QX, verbose=False)
QP  = np.argmax(QP2, axis=1)
cm  = confusion_matrix(QY, QP)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_title('Matriz de Confusão — CNN LeNet/MNIST')
ax.set_xlabel('Classe Predita')
ax.set_ylabel('Classe Verdadeira')
ax.set_xticks(range(nclasses))
ax.set_yticks(range(nclasses))
limiar = cm.max() / 2
for i in range(nclasses):
    for j in range(nclasses):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                color='white' if cm[i, j] > limiar else 'black')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS, 'matriz_confusao.png'), dpi=150)
plt.close()
print('Matriz de confusão salva: resultados/matriz_confusao.png')

# =============================================================================
# Resumo
# =============================================================================
nerro_pretrained = int(np.count_nonzero(np.argmax(
    model.predict(QX_inv, verbose=False), axis=1) - QY))
nerro_novo       = int(np.count_nonzero(QP - QY))

print('\n========================================')
print('RESUMO — ENTREGA 11')
print('========================================')
print(f'Modelo pré-treinado (cnn3.keras):')
print(f'  Acurácia: {score_pretrained[1]*100:.2f}%  '
      f'Erro: {(1-score_pretrained[1])*100:.2f}%  '
      f'({nerro_pretrained} erros)')
print(f'Rede treinada do zero (30 épocas):')
print(f'  Acurácia: {score_novo[1]*100:.2f}%  '
      f'Erro: {(1-score_novo[1])*100:.2f}%  '
      f'({nerro_novo} erros)')
print(f'  Tempo de treino: {t1-t0:.2f}s')
print('========================================')
