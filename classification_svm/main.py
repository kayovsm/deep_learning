import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HeartDiseaseClassifier:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.selected_features = None
        
    def load_and_preprocess_data(self, file_path):
        print("=== CARREGANDO E PREPROCESSANDO DADOS ===")
        
        # carregando dados do arquivo CSV
        self.data = pd.read_csv(file_path)
        print(f"Dados carregados: {self.data.shape[0]} linhas, {self.data.shape[1]} colunas")
        
        # verificando valores faltantes
        na_counts = self.data.isnull().sum()
        print(f"Valores NA por coluna:\n{na_counts[na_counts > 0]}")
        
        # removendo linhas com valores NA
        initial_rows = self.data.shape[0]
        self.data = self.data.dropna()
        print(f"Linhas removidas (NA): {initial_rows - self.data.shape[0]}")
        print(f"Dados finais: {self.data.shape[0]} linhas")
        
        target_dist = self.data['AHD'].value_counts()
        print(f"\nDistribuição da variável target (AHD):\n{target_dist}")
        print(f"Proporção: {target_dist / len(self.data) * 100}")
        
        return self.data
    
    def select_features(self, feature1='Age', feature2='MaxHR'):
        print(f"\n=== SELECIONANDO FEATURES ===")
        print(f"Features selecionadas: {feature1}, {feature2}")
        
        # definindo as 2 features escolhidas
        self.selected_features = [feature1, feature2]
        
        # matriz de features (X)
        self.X = self.data[self.selected_features].values
        
        # codificando variavel target (Yes/No -> 1/0)
        le = LabelEncoder()
        self.y = le.fit_transform(self.data['AHD'])
        
        print(f"Shape das features: {self.X.shape}")
        print(f"Classes: {le.classes_} -> {np.unique(self.y)}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.3, random_state=42):
        print(f"\n=== DIVIDINDO DADOS (TREINO: {1-test_size:.0%}, TESTE: {test_size:.0%}) ===")
        
        # divisao estratificada para manter balanceamento
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # normalizacao dos dados (StandardScaler)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Treino: {len(self.X_train)} amostras")
        print(f"Teste: {len(self.X_test)} amostras")
        print(f"Distribuição treino: {np.bincount(self.y_train)}")
        print(f"Distribuição teste: {np.bincount(self.y_test)}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def plot_data_distribution(self):
        print("\n=== CRIANDO VISUALIZAÇÕES DOS DADOS ===")
        
        # criando subplots para treino e teste
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # cores e labels para as classes
        colors = ['red', 'blue']
        labels = ['No Disease', 'Disease']
        
        # grafico do conjunto de treino
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = self.y_train == i
            axes[0].scatter(self.X_train[mask, 0], self.X_train[mask, 1], 
                           c=color, alpha=0.6, label=label, s=50)
        
        axes[0].set_xlabel(self.selected_features[0])
        axes[0].set_ylabel(self.selected_features[1])
        axes[0].set_title('Conjunto de Treino')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # grafico do conjunto de teste
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = self.y_test == i
            axes[1].scatter(self.X_test[mask, 0], self.X_test[mask, 1], 
                           c=color, alpha=0.6, label=label, s=50)
        
        axes[1].set_xlabel(self.selected_features[0])
        axes[1].set_ylabel(self.selected_features[1])
        axes[1].set_title('Conjunto de Teste')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def train_svm_models(self):
        print("\n=== TREINANDO MODELOS SVM ===")
        
        # parametros para otimizacao de cada kernel
        param_grids = {
            'linear': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear']
            },
            'poly': {
                'C': [0.1, 1, 10],
                'kernel': ['poly'],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto']
            },
            'rbf': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        }
        
        # dicionario para armazenar os modelos treinados
        self.models = {}
        
        # treinando cada tipo de kernel SVM
        for kernel_name, param_grid in param_grids.items():
            print(f"\nTreinando SVM {kernel_name.upper()}...")
            
            # grid search para encontrar melhores hiperparametros
            svm = SVC(random_state=42)
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            self.models[kernel_name] = grid_search.best_estimator_
            
            print(f"Melhores parâmetros: {grid_search.best_params_}")
            print(f"Score CV: {grid_search.best_score_:.3f}")
    
    def plot_decision_boundaries(self, dataset='train'):
        # escolhendo conjunto de dados para plotar
        if dataset == 'train':
            X_plot = self.X_train_scaled
            y_plot = self.y_train
            title_suffix = "Treino"
        else:
            X_plot = self.X_test_scaled
            y_plot = self.y_test
            title_suffix = "Teste"
        
        print(f"\n=== PLOTANDO FRONTEIRAS DE DECISÃO ({title_suffix.upper()}) ===")
        
        # criando subplots para os 3 kernels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # criando grid para plotar regioes de decisao
        h = 0.02
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        colors = ['red', 'blue']
        labels = ['No Disease', 'Disease']
        
        # plotando para cada kernel SVM
        for idx, (kernel_name, model) in enumerate(self.models.items()):
            ax = axes[idx]
            
            # predicao para todos os pontos do grid
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # plotando regioes de decisao
            ax.contourf(xx, yy, Z, alpha=0.3, colors=['lightcoral', 'lightblue'])
            ax.contour(xx, yy, Z, colors='black', linestyles='--', linewidths=1)
            
            for i, (color, label) in enumerate(zip(colors, labels)):
                mask = y_plot == i
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                          c=color, alpha=0.8, label=label, s=50, edgecolors='black')
            
            if hasattr(model, 'support_vectors_'):
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                          s=100, facecolors='none', edgecolors='green', linewidths=2,
                          label='Support Vectors')
            
            ax.set_xlabel(f'{self.selected_features[0]} (normalizado)')
            ax.set_ylabel(f'{self.selected_features[1]} (normalizado)')
            ax.set_title(f'SVM {kernel_name.upper()} - {title_suffix}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_support_vectors_and_margins(self):
        print("\n=== PLOTANDO VETORES DE SUPORTE E MARGENS ===")
        
        # criando subplots para os 3 kernels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = ['red', 'blue']
        labels = ['No Disease', 'Disease']
        
        # plotando vetores de suporte para cada kernel
        for idx, (kernel_name, model) in enumerate(self.models.items()):
            ax = axes[idx]
            
            for i, (color, label) in enumerate(zip(colors, labels)):
                mask = self.y_train == i
                ax.scatter(self.X_train_scaled[mask, 0], self.X_train_scaled[mask, 1], 
                          c=color, alpha=0.6, label=label, s=50)
            
            if hasattr(model, 'support_vectors_'):
                ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                          s=200, facecolors='none', edgecolors='green', linewidths=3,
                          label=f'Support Vectors ({len(model.support_vectors_)})')
                
                # para kernel linear, plotar margem
                if kernel_name == 'linear':
                    w = model.coef_[0]
                    b = model.intercept_[0]
                    
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    
                    # linha de decisao
                    xx = np.linspace(xlim[0], xlim[1], 30)
                    yy = -(w[0] * xx + b) / w[1]
                    
                    # calculando margem
                    margin = 1 / np.sqrt(np.sum(w ** 2))
                    yy_down = yy - np.sqrt(1 + (w[0]/w[1])**2) * margin
                    yy_up = yy + np.sqrt(1 + (w[0]/w[1])**2) * margin
                    
                    ax.plot(xx, yy, 'k-', label='Decision Boundary')
                    ax.plot(xx, yy_down, 'k--', alpha=0.5, label='Margin')
                    ax.plot(xx, yy_up, 'k--', alpha=0.5)
                    
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            
            ax.set_xlabel(f'{self.selected_features[0]} (normalizado)')
            ax.set_ylabel(f'{self.selected_features[1]} (normalizado)')
            ax.set_title(f'SVM {kernel_name.upper()} - Vetores de Suporte')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_models(self):
        print("\n=== AVALIAÇÃO DOS MODELOS ===")
        
        # dicionario para armazenar resultados
        results = {}
        
        # avaliando cada modelo SVM
        for kernel_name, model in self.models.items():
            # predicoes nos conjuntos de treino e teste
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # calculando acuracias
            train_acc = accuracy_score(self.y_train, y_pred_train)
            test_acc = accuracy_score(self.y_test, y_pred_test)
            
            results[kernel_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'y_pred_test': y_pred_test
            }
            
            print(f"\n{kernel_name.upper()} SVM:")
            print(f"  Acurácia Treino: {train_acc:.3f} ({train_acc*100:.1f}%)")
            print(f"  Acurácia Teste:  {test_acc:.3f} ({test_acc*100:.1f}%)")
            
            print(f"\nRelatório de Classificação - {kernel_name.upper()}:")
            print(classification_report(self.y_test, y_pred_test, 
                                      target_names=['No Disease', 'Disease']))
        
        return results
    
    def plot_accuracy_comparison(self, results):
        kernels = list(results.keys())
        train_accs = [results[k]['train_accuracy'] for k in kernels]
        test_accs = [results[k]['test_accuracy'] for k in kernels]
        
        x = np.arange(len(kernels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, train_accs, width, label='Treino', alpha=0.8)
        ax.bar(x + width/2, test_accs, width, label='Teste', alpha=0.8)
        
        ax.set_xlabel('Kernel SVM')
        ax.set_ylabel('Acurácia')
        ax.set_title('Comparação de Acurácias por Kernel')
        ax.set_xticks(x)
        ax.set_xticklabels([k.upper() for k in kernels])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
            ax.text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}', 
                   ha='center', va='bottom')
            ax.text(i + width/2, test_acc + 0.01, f'{test_acc:.3f}', 
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def train_full_feature_model(self, results):
        print("\n=== MODELO COM TODAS AS FEATURES (OPCIONAL) ===")
        
        # selecionando todas as features numericas
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['Unnamed: 0']]
        
        X_full = self.data[feature_columns].values
        
        # codificando variaveis categoricas
        categorical_columns = ['ChestPain', 'Thal']
        for col in categorical_columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].fillna('unknown'))
                feature_columns.append(f'{col}_encoded')
        
        X_full = self.data[feature_columns].values
        
        # divisao treino/teste com todas as features
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X_full, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # normalizacao dos dados
        scaler_full = StandardScaler()
        X_train_full_scaled = scaler_full.fit_transform(X_train_full)
        X_test_full_scaled = scaler_full.transform(X_test_full)
        
        # treinando SVM RBF com todas as features
        svm_full = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        svm_full.fit(X_train_full_scaled, y_train_full)
        
        # avaliando o modelo
        y_pred_full = svm_full.predict(X_test_full_scaled)
        acc_full = accuracy_score(y_test_full, y_pred_full)
        
        print(f"Features utilizadas: {len(feature_columns)}")
        print(f"Acurácia com todas as features: {acc_full:.3f} ({acc_full*100:.1f}%)")
        print(f"\nComparação:")
        print(f"  2 features: {max([results[k]['test_accuracy'] for k in results.keys()]):.3f}")
        print(f"  {len(feature_columns)} features: {acc_full:.3f}")
        
        return acc_full


def main():
    print("CLASSIFICACAO SVM - PREDICAO DE DOENCA CARDIACA")
    print("="*60)
    
    classifier = HeartDiseaseClassifier()
    
    classifier.load_and_preprocess_data('heart.csv')
    classifier.select_features('Age', 'MaxHR')
    classifier.split_data()
    classifier.plot_data_distribution()
    classifier.train_svm_models()
    classifier.plot_support_vectors_and_margins()
    classifier.plot_decision_boundaries('train')
    results = classifier.evaluate_models()
    classifier.plot_decision_boundaries('test')
    classifier.plot_accuracy_comparison(results)
    classifier.train_full_feature_model(results)


if __name__ == "__main__":
    main()
