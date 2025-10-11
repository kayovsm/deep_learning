import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import joblib
import warnings
warnings.filterwarnings('ignore')

# CONFIGURAÇÕES PRINCIPAIS
IMAGE_SIZE = (64, 64)         # largura, altura em pixels
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
TEST_SIZE = 0.3
RANDOM_STATE = 42
POS_DIR = 'positive'
NEG_DIR = 'negative'
OUT_MODEL = 'best_svm.joblib'


# carrega imagens de uma pasta, converte para tons de cinza e redimensiona
def load_images_from_folder(folder, label, image_size=IMAGE_SIZE):

    imgs = []
    labels = []
    filenames = []

    # se a pasta nao existir, retornamos listas vazias
    if not os.path.exists(folder):
        return imgs, labels, filenames

    # ordena os nomes dos arquivos para comportamento determinístico
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(folder, fname)
        img = io.imread(path)
        if img is None:
            continue
    # converte para tons de cinza quando necessário
        if img.ndim == 3:
            img = rgb2gray(img)
        img_resized = resize(img, image_size, anti_aliasing=True)
        imgs.append(img_resized)
        labels.append(label)
        filenames.append(path)
    return imgs, labels, filenames


# calcula HOG para uma lista de imagens e retorna matriz (n_samples, n_features)
def compute_hog_features(images, hog_params=HOG_PARAMS):
    features = []
    for im in images:
        f = hog(im, **hog_params)
        features.append(f)
    return np.array(features)


def main():
    print('=== HOG + SVM PIPELINE ===')
    print(f'PASTAS ESPERADAS: {POS_DIR}  {NEG_DIR}')

    # carrega imagens
    pos_imgs, pos_labels, pos_files = load_images_from_folder(POS_DIR, 1)
    neg_imgs, neg_labels, neg_files = load_images_from_folder(NEG_DIR, 0)

    # verifica se as pastas existem e têm imagens
    if len(pos_imgs) == 0 or len(neg_imgs) == 0:
        print('\nErro: pastas ou imagens não encontradas.')
        print('Crie pastas `positive/` e `negative/` com imagens e rode novamente.')
        return

    images = np.array(pos_imgs + neg_imgs)
    labels = np.array(pos_labels + neg_labels)
    files = np.array(pos_files + neg_files)

    print(f'Total imagens: {len(images)} (Pos: {np.sum(labels==1)}, Neg: {np.sum(labels==0)})')

    # extrai descritores HOG para todas as imagens
    print('\n=== EXTRAINDO HOG ===')
    print('Computando HOG para todas as imagens...')
    X = compute_hog_features(images)
    y = labels
    print('HOG feature shape:', X.shape)

    # separa os dados em treino e teste mantendo a proporção das classes
    X_train, X_test, y_train, y_test, files_train, files_test, imgs_train, imgs_test = train_test_split(
        X, y, files, images, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print(f'-- Train: {len(X_train)}, Test: {len(X_test)}')
    print(f'DISTRIBUICAO TREINO: {np.bincount(y_train)}')
    print(f'DISTRIBUICAO TESTE:  {np.bincount(y_test)}')

    # standardização das features (zero mean, unit variance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # grid search
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }

    print('\n=== TREINANDO SVM (GridSearchCV) ===')
    print('Rodando GridSearchCV (pode demorar alguns minutos)')

    # cria o classificador SVM com GridSearchCV
    svm = SVC(random_state=RANDOM_STATE)
    grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print('Best params:', grid.best_params_)
    best = grid.best_estimator_

    # SALVA MODELO E SCALER
    joblib.dump({'model': best, 'scaler': scaler, 'hog_params': HOG_PARAMS}, OUT_MODEL)
    print('MODELO SALVO EM:', OUT_MODEL)

    # avaliação
    y_pred_train = best.predict(X_train_scaled)
    y_pred_test = best.predict(X_test_scaled)

    print('\n=== AVALIACAO ===')
    print('ACURACIA TREINO:', accuracy_score(y_train, y_pred_train))
    print('ACURACIA TESTE: ', accuracy_score(y_test, y_pred_test))
    print('\nClassification report (test):')
    print(classification_report(y_test, y_pred_test, target_names=['neg','pos']))

    cm = confusion_matrix(y_test, y_pred_test)
    print('\nConfusion matrix:\n', cm)

    # exemplos qualitativos
    tp = np.where((y_test == 1) & (y_pred_test == 1))[0]
    tn = np.where((y_test == 0) & (y_pred_test == 0))[0]
    fp = np.where((y_test == 0) & (y_pred_test == 1))[0]
    fn = np.where((y_test == 1) & (y_pred_test == 0))[0]

    def plot_examples(indices, title_prefix):
        # mostra até 3 exemplos por categoria
        n = min(3, len(indices))
        if n == 0:
            print(f'Nenhum exemplo para {title_prefix}')
            return
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        for i, idx in enumerate(indices[:n]):
            ax = axes[i] if n > 1 else axes
            ax.imshow(imgs_test[idx], cmap='gray')
            ax.set_title(f"{title_prefix}\n{os.path.basename(files_test[idx])}")
            ax.axis('off')
        plt.show()

    plot_examples(tp, 'True Positive')
    plot_examples(tn, 'True Negative')
    plot_examples(fp, 'False Positive')
    plot_examples(fn, 'False Negative')


if __name__ == '__main__':
    main()
