#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tensorflow",
#   "matplotlib",
#   "scikit-learn",
#   "pillow",
# ]
# ///
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from time import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, Resizing,
)
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0

tf.random.set_seed(42)
np.random.seed(42)

RAIZ       = os.path.dirname(os.path.abspath(__file__))
RESULTADOS = os.path.join(RAIZ, 'resultados')

CLASSES = ['avião', 'automóvel', 'pássaro', 'gato', 'veado',
           'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']
NCLASSES    = 10
BATCH_SIZE  = 64
EPOCHS_BL   = 30
EPOCHS_EFF1 = 5
EPOCHS_EFF2 = 10

# =============================================================================
# 1. Explorar CIFAR-10
# =============================================================================
print('=== 1. Carregando e explorando CIFAR-10 ===')
(ax_raw, ay_raw), (qx_raw, qy_raw) = cifar10.load_data()
ay = ay_raw.flatten()
qy = qy_raw.flatten()
print(f'Treino: {ax_raw.shape}  |  Teste: {qx_raw.shape}')

contagem_treino = [int(np.sum(ay == i)) for i in range(NCLASSES)]
contagem_teste  = [int(np.sum(qy == i)) for i in range(NCLASSES)]
print('Distribuição treino:', dict(zip(CLASSES, contagem_treino)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.bar(CLASSES, contagem_treino, color='steelblue')
ax1.set_title('Distribuição das Classes — Treino')
ax1.set_xlabel('Classe')
ax1.set_ylabel('Quantidade')
ax1.tick_params(axis='x', rotation=45)

ax2.bar(CLASSES, contagem_teste, color='darkorange')
ax2.set_title('Distribuição das Classes — Teste')
ax2.set_xlabel('Classe')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS, 'distribuicao_classes.png'), dpi=150)
plt.close()
print('Distribuição salva: resultados/distribuicao_classes.png')

fig, axes = plt.subplots(NCLASSES, 10, figsize=(15, 15))
for cls in range(NCLASSES):
    idxs = np.where(qy == cls)[0][:10]
    for col, idx in enumerate(idxs):
        axes[cls, col].imshow(qx_raw[idx])
        axes[cls, col].axis('off')
    axes[cls, 0].set_ylabel(CLASSES[cls], rotation=0, fontsize=8,
                             labelpad=55, va='center')
plt.suptitle('Exemplos do CIFAR-10 — 10 amostras por classe', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS, 'exemplos_cifar10.png'), dpi=150)
plt.close()
print('Exemplos salvos: resultados/exemplos_cifar10.png')

# =============================================================================
# Pré-processamento
# =============================================================================
ax_norm = ax_raw.astype('float32') / 255.0 - 0.5
qx_norm = qx_raw.astype('float32') / 255.0 - 0.5
ay2 = keras.utils.to_categorical(ay, NCLASSES)
qy2 = keras.utils.to_categorical(qy, NCLASSES)

ax_eff = ax_raw.astype('float32')  # [0, 255] para EfficientNet
qx_eff = qx_raw.astype('float32')

INPUT_SHAPE = (32, 32, 3)


def salva_historico(history, nome_arq, titulo):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history.history['accuracy'],     label='treino')
    if 'val_accuracy' in history.history:
        a1.plot(history.history['val_accuracy'], label='validação')
    a1.set_title(f'Acurácia — {titulo}')
    a1.set_xlabel('Época')
    a1.set_ylabel('Acurácia')
    a1.legend(loc='lower right')

    a2.plot(history.history['loss'],     label='treino')
    if 'val_loss' in history.history:
        a2.plot(history.history['val_loss'], label='validação')
    a2.set_title(f'Perda — {titulo}')
    a2.set_xlabel('Época')
    a2.set_ylabel('Perda')
    a2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTADOS, nome_arq), dpi=150)
    plt.close()

# =============================================================================
# 2. Baseline — rede "tipo LeNet" (sem pré-treinamento)
# =============================================================================
print('\n=== 2. Baseline "tipo LeNet" ===')
baseline = Sequential([
    Conv2D(20, kernel_size=(5, 5), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(40, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.25),
    Dense(1000, activation='relu'),
    Dropout(0.25),
    Dense(NCLASSES, activation='softmax'),
], name='baseline_lenet')
baseline.summary()

baseline.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
ckpt_bl = ModelCheckpoint(
    os.path.join(RESULTADOS, 'baseline.keras'), verbose=0, save_best_only=True
)

t0 = time()
hist_bl = baseline.fit(
    ax_norm, ay2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_BL,
    verbose=2,
    validation_split=0.2,
    callbacks=[ckpt_bl],
)
t1 = time()
tempo_bl = t1 - t0
print(f'Tempo de treino: {tempo_bl:.2f}s')

baseline = keras.models.load_model(os.path.join(RESULTADOS, 'baseline.keras'))
score_bl = baseline.evaluate(qx_norm, qy2, verbose=False)
print(f'Acurácia baseline: {score_bl[1]*100:.2f}%')
salva_historico(hist_bl, 'curva_baseline.png', 'Baseline LeNet')
print('Curva salva: resultados/curva_baseline.png')

# =============================================================================
# 3. Fine-tuning com EfficientNetB0
# =============================================================================
print('\n=== 3. Fine-tuning com EfficientNetB0 ===')

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(96, 96, 3),
)
base_model.trainable = False

inputs = keras.Input(shape=INPUT_SHAPE)
x = Resizing(96, 96)(inputs)           # CIFAR-10 32×32 → 96×96
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(NCLASSES, activation='softmax')(x)
model_eff = Model(inputs, outputs, name='efficientnet_cifar')

model_eff.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model_eff.summary()

print('Fase 1: treinando cabeça (base congelada)...')
t0 = time()
hist_eff1 = model_eff.fit(
    ax_eff, ay2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_EFF1,
    verbose=2,
    validation_split=0.2,
)
t1 = time()
print(f'Tempo fase 1: {t1-t0:.2f}s')

print('\nFase 2: fine-tuning últimas camadas...')
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model_eff.compile(
    optimizer=optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
ckpt_eff = ModelCheckpoint(
    os.path.join(RESULTADOS, 'efficientnet_finetuned.keras'), verbose=0, save_best_only=True
)

t0 = time()
hist_eff2 = model_eff.fit(
    ax_eff, ay2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_EFF2,
    verbose=2,
    validation_split=0.2,
    callbacks=[ckpt_eff],
)
t1 = time()
tempo_eff = t1 - t0
print(f'Tempo fase 2: {tempo_eff:.2f}s')

model_eff = keras.models.load_model(os.path.join(RESULTADOS, 'efficientnet_finetuned.keras'))
score_eff = model_eff.evaluate(qx_eff, qy2, verbose=False)
print(f'Acurácia EfficientNetB0 fine-tuned: {score_eff[1]*100:.2f}%')

acc_combined     = hist_eff1.history['accuracy']     + hist_eff2.history['accuracy']
val_acc_combined = hist_eff1.history['val_accuracy'] + hist_eff2.history['val_accuracy']
loss_combined    = hist_eff1.history['loss']         + hist_eff2.history['loss']
val_loss_combined= hist_eff1.history['val_loss']     + hist_eff2.history['val_loss']

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(acc_combined,     label='treino')
a1.plot(val_acc_combined, label='validação')
a1.axvline(x=EPOCHS_EFF1, color='gray', linestyle='--', label='início fine-tuning')
a1.set_title('Acurácia — EfficientNetB0')
a1.set_xlabel('Época')
a1.set_ylabel('Acurácia')
a1.legend(loc='lower right')

a2.plot(loss_combined,     label='treino')
a2.plot(val_loss_combined, label='validação')
a2.axvline(x=EPOCHS_EFF1, color='gray', linestyle='--', label='início fine-tuning')
a2.set_title('Perda — EfficientNetB0')
a2.set_xlabel('Época')
a2.set_ylabel('Perda')
a2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS, 'curva_efficientnet.png'), dpi=150)
plt.close()
print('Curva EfficientNet salva: resultados/curva_efficientnet.png')

# =============================================================================
# 4. Baseline com data augmentation
# =============================================================================
print('\n=== 4. Baseline com data augmentation ===')
baseline_da = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.1),
    Conv2D(20, kernel_size=(5, 5), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(40, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.25),
    Dense(1000, activation='relu'),
    Dropout(0.25),
    Dense(NCLASSES, activation='softmax'),
], name='baseline_lenet_da')

baseline_da.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
ckpt_da = ModelCheckpoint(
    os.path.join(RESULTADOS, 'baseline_da.keras'), verbose=0, save_best_only=True
)

t0 = time()
hist_da = baseline_da.fit(
    ax_norm, ay2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_BL,
    verbose=2,
    validation_split=0.2,
    callbacks=[ckpt_da],
)
t1 = time()
tempo_da = t1 - t0
print(f'Tempo de treino com DA: {tempo_da:.2f}s')

baseline_da = keras.models.load_model(os.path.join(RESULTADOS, 'baseline_da.keras'))
score_da = baseline_da.evaluate(qx_norm, qy2, verbose=False)
print(f'Acurácia com data augmentation: {score_da[1]*100:.2f}%')
salva_historico(hist_da, 'curva_baseline_da.png', 'Baseline + Data Augmentation')
print('Curva DA salva: resultados/curva_baseline_da.png')

# =============================================================================
# 5. Comparação final
# =============================================================================
print('\n=== 5. Comparação final ===')
rotulos_finais   = ['Baseline\n(sem DA)', 'Baseline\n(com DA)', 'EfficientNetB0\n(fine-tuning)']
acuracias_finais = [score_bl[1]*100, score_da[1]*100, score_eff[1]*100]
cores = ['steelblue', 'mediumseagreen', 'darkorange']

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(rotulos_finais, acuracias_finais, color=cores, width=0.5)
ax.set_ylim(0, 105)
ax.set_title('Comparação Final — CIFAR-10')
ax.set_ylabel('Acurácia (%)')
for bar, acc in zip(bars, acuracias_finais):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTADOS, 'comparacao_final.png'), dpi=150)
plt.close()
print('Comparação final salva: resultados/comparacao_final.png')

# =============================================================================
# Resumo
# =============================================================================
print('\n========================================')
print('RESUMO — ENTREGA 12')
print('========================================')
print(f'Baseline "tipo LeNet" (sem DA, {EPOCHS_BL} épocas):')
print(f'  Acurácia: {score_bl[1]*100:.2f}%  |  Tempo: {tempo_bl:.0f}s')
print(f'Baseline com data augmentation ({EPOCHS_BL} épocas):')
print(f'  Acurácia: {score_da[1]*100:.2f}%  |  Ganho: '
      f'+{(score_da[1]-score_bl[1])*100:.2f}pp')
print(f'EfficientNetB0 fine-tuning ({EPOCHS_EFF1}+{EPOCHS_EFF2} épocas, 32→96px):')
print(f'  Acurácia: {score_eff[1]*100:.2f}%  |  Ganho vs. baseline: '
      f'+{(score_eff[1]-score_bl[1])*100:.2f}pp')
print('========================================')
