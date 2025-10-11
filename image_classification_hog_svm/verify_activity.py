import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# configuração de caminhos e parâmetros (sincronizados com main.py)
ROOT = os.path.dirname(__file__)
POS_DIR = os.path.join(ROOT, 'positive')
NEG_DIR = os.path.join(ROOT, 'negative')
MODEL_PATH = os.path.join(ROOT, 'best_svm.joblib')
OUT_DIR = os.path.join(ROOT, 'out')
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE = (64,64)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')


def load_images(folder):
    # carrega imagens da pasta, converte para grayscale e redimensiona
    imgs = []
    files = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        p = os.path.join(folder, f)
        img = imread(p)
        if img.ndim==3:
            img = rgb2gray(img)
        img = resize(img, IMAGE_SIZE, anti_aliasing=True)
        imgs.append(img)
        files.append(p)
    return imgs, files


pos_imgs, pos_files = load_images(POS_DIR)
neg_imgs, neg_files = load_images(NEG_DIR)

print('POS:', len(pos_imgs), 'NEG:', len(neg_imgs))

images = np.array(pos_imgs + neg_imgs)
labels = np.array([1]*len(pos_imgs) + [0]*len(neg_imgs))
files = np.array(pos_files + neg_files)

# calcula descritores HOG para todas as imagens
X = np.array([hog(im, **HOG_PARAMS) for im in images])
print('X SHAPE:', X.shape)

# separa os dados em treino e teste (70/30 estratificado)
X_train, X_test, y_train, y_test, files_train, files_test, imgs_train, imgs_test = train_test_split(
    X, labels, files, images, test_size=0.3, random_state=42, stratify=labels)
print('TRAIN/TEST SIZES:', len(X_train), len(X_test))
print('DISTRIBUICAO TREINO:', np.bincount(y_train), 'DISTRIBUICAO TESTE:', np.bincount(y_test))

# padroniza as features com StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# carrega o modelo salvo (pode ser um dicionário com modelo e scaler)
obj = joblib.load(MODEL_PATH)
if isinstance(obj, dict):
    model = obj.get('model', None)
    saved_scaler = obj.get('scaler', None)
    if saved_scaler is not None:
        scaler = saved_scaler
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
else:
    model = obj

print('MODELO CARREGADO:', type(model))

# avalia o desempenho do modelo em treino e teste
y_pred_train = model.predict(X_train_s)
y_pred_test = model.predict(X_test_s)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
print('ACURACIA TREINO:', train_acc, 'ACURACIA TESTE:', test_acc)
print('\nRELATORIO (TESTE):')
print(classification_report(y_test, y_pred_test, target_names=['neg','pos']))
print('MATRIZ DE CONFUSAO:\n', confusion_matrix(y_test, y_pred_test))

# salva a figura da matriz de confusão
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(4,4))
ax.matshow(cm, cmap='Blues')
for (i,j), val in np.ndenumerate(cm):
    ax.text(j, i, str(val), ha='center', va='center')
ax.set_xticks([0,1]); ax.set_yticks([0,1]);
ax.set_xticklabels(['neg','pos']); ax.set_yticklabels(['neg','pos']);
ax.set_xlabel('Pred'); ax.set_ylabel('True');
fig.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'))
plt.close(fig)

# gera mosaicos com índices de TP/TN/FP/FN para inspeção manual
y = y_test
yp = y_pred_test
indices = dict(
    tp = np.where((y==1)&(yp==1))[0],
    tn = np.where((y==0)&(yp==0))[0],
    fp = np.where((y==0)&(yp==1))[0],
    fn = np.where((y==1)&(yp==0))[0]
)

for cat, idxs in indices.items():
    n = min(3, len(idxs))
    fig, axes = plt.subplots(1, n if n>0 else 1, figsize=(4*n,4))
    if n==0:
        axes.text(0.5,0.5,'Sem exemplos disponíveis', ha='center', va='center')
        axes.axis('off')
    else:
        for i, idx in enumerate(idxs[:n]):
            ax = axes[i] if n>1 else axes
            ax.imshow(imgs_test[idx], cmap='gray')
            ax.set_title(f'{cat} {os.path.basename(files_test[idx])}')
            ax.axis('off')
    fig.savefig(os.path.join(OUT_DIR, f'{cat}_examples.png'))
    plt.close(fig)

print('Resultados salvos em', OUT_DIR)
