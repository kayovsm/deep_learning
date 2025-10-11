import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = os.path.dirname(__file__)
POS_DIR = os.path.join(ROOT, 'positive')
NEG_DIR = os.path.join(ROOT, 'negative')
MODEL_PATH = os.path.join(ROOT, 'best_svm.joblib')

IMAGE_SIZE = (64,64)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')


def load_images(folder):
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

images = np.array(pos_imgs + neg_imgs)
labels = np.array([1]*len(pos_imgs) + [0]*len(neg_imgs))
files = np.array(pos_files + neg_files)

# calcula descritores HOG para cada imagem
X = np.array([hog(im, **HOG_PARAMS) for im in images])

# separa em treino e teste (determinístico quando random_state é fixado)
X_train, X_test, y_train, y_test, files_train, files_test, imgs_train, imgs_test = train_test_split(
    X, labels, files, images, test_size=0.3, random_state=42, stratify=labels)

# normaliza as features usando StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# carrega o modelo salvo (e scaler, se estiver empacotado)
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

# realiza as predições no conjunto de teste
y_pred_test = model.predict(X_test_s)

# seleciona índices para cada categoria de acerto/erro
y = y_test
yp = y_pred_test
indices = dict(
    tp = np.where((y==1)&(yp==1))[0],
    tn = np.where((y==0)&(yp==0))[0],
    fp = np.where((y==0)&(yp==1))[0],
    fn = np.where((y==1)&(yp==0))[0]
)

# imprime quantidade e nomes dos arquivos por categoria (limitado a 50)
for cat, idxs in indices.items():
    print(f"\n=== {cat.upper()} (count={len(idxs)}) ===")
    if len(idxs)==0:
        print('  (none)')
        continue
    # print up to 50 entries
    for i, idx in enumerate(idxs[:50]):
        print(f"  {i+1:2d}. {os.path.basename(files_test[idx])} -> true={y[idx]} pred={yp[idx]}")

print('\nPRONTO')
