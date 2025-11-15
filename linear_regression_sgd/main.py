import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATASET_PATH = "../linear_regression/Income1.csv"
LEARNING_RATE = 0.1
EPOCHS = 1000
BATCH_SIZE = 8
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# implementando o algoritmo Stochastic Gradient Descent com minibatches
class LinearRegressionSGD:
    def __init__(self, learning_rate=0.001, epochs=1000, batch_size=8):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.theta = None
        self.cost_history = []
        self.features_mean = None
        self.features_std = None
        self.target_mean = None
        self.target_std = None
        
    def _compute_cost(self, features, target):
        # calculando o MSE
        num_samples = len(target)
        predictions = features @ self.theta
        cost = (1 / (2 * num_samples)) * np.sum((predictions - target) ** 2)
        return cost
    
    def fit(self, features, target):
        # normalizando os dados
        self.features_mean = np.mean(features, axis=0)
        self.features_std = np.std(features, axis=0)
        self.target_mean = np.mean(target)
        self.target_std = np.std(target)
        
        features_normalized = (features - self.features_mean) / self.features_std
        target_normalized = (target - self.target_mean) / self.target_std
        
        # adicionando coluna de intercept
        num_samples, num_features = features_normalized.shape
        features_with_bias = np.c_[np.ones((num_samples, 1)), features_normalized]
        
        # inicializando os pesos com valores aleatorios pequenos
        self.theta = np.random.randn(num_features + 1, 1) * 0.01
        
        print(f"___TREINAMENTO SGD___")
        print(f"Amostras: {num_samples}")
        print(f"Features: {num_features}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        
        # algoritmo SGD com minibatches
        for epoch in range(self.epochs):
            # embaralhando os dados a cada epoca
            shuffled_indices = np.random.permutation(num_samples)
            features_shuffled = features_with_bias[shuffled_indices]
            target_shuffled = target_normalized[shuffled_indices].reshape(-1, 1)
            
            # dividindo em minibatches
            for batch_start_idx in range(0, num_samples, self.batch_size):
                # selecionando o minibatch
                batch_end_idx = batch_start_idx + self.batch_size
                features_batch = features_shuffled[batch_start_idx:batch_end_idx]
                target_batch = target_shuffled[batch_start_idx:batch_end_idx]
                
                # calculando o gradiente para o minibatch
                current_batch_size = len(features_batch)
                batch_predictions = features_batch @ self.theta
                gradient = (1 / current_batch_size) * features_batch.T @ (batch_predictions - target_batch)
                
                # atualizando os pesos
                self.theta -= self.learning_rate * gradient
            
            # calculando e armazenando o custo a cada 10 epocas
            if epoch % 10 == 0:
                cost = self._compute_cost(features_with_bias, target_normalized.reshape(-1, 1))
                self.cost_history.append(cost)
                
                if epoch % 100 == 0:
                    print(f"Época {epoch:4d} | MSE (normalizado): {cost:.4f}")
        
        # custo final
        final_cost = self._compute_cost(features_with_bias, target_normalized.reshape(-1, 1))
        self.cost_history.append(final_cost)
        
        # desnormalizando os coeficientes para a escala original
        theta_denormalized = self.theta.copy()
        theta_denormalized[1:] = self.theta[1:] * (self.target_std / self.features_std.reshape(-1, 1))
        theta_denormalized[0] = self.target_mean - np.sum(theta_denormalized[1:] * self.features_mean.reshape(-1, 1))
        self.theta_original = theta_denormalized
        
        print(f"\n___TREINAMENTO CONCLUIDO___")
        print(f"MSE final (normalizado): {final_cost:.4f}")
        print(f"Theta (intercept): {self.theta_original[0, 0]:.4f}")
        print(f"Theta (slope): {self.theta_original[1, 0]:.4f}")
        
        return self
    
    def predict(self, features):
        # adicionando coluna de bias e fazendo a prediçao na escala original
        features_with_bias = np.c_[np.ones((features.shape[0], 1)), features]
        return features_with_bias @ self.theta_original
    
    def get_params(self):
        # retornando intercept e slope na escala original
        return self.theta_original[0, 0], self.theta_original[1, 0]


def main():
    print("___REGRESSÃO LINEAR COM SGD___\n")
    
    # carregando os dados do arquivo csv
    dataset = pd.read_csv(DATASET_PATH, index_col=0)
    print(f"Dataset carregado: {dataset.shape[0]} linhas, {dataset.shape[1]} colunas")
    
    education_years = dataset["Education"].values
    income_values = dataset["Income"].values
    
    # fazendo reshape para formato correto
    features = education_years.reshape(-1, 1)
    target = income_values
    
    # criando e treinando o modelo SGD
    print()
    model_sgd = LinearRegressionSGD(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    model_sgd.fit(features, target)
    
    # obtendo os parametros do modelo SGD
    intercept_sgd, slope_sgd = model_sgd.get_params()
    
    # calculando MSE do SGD na escala original
    predictions_sgd = intercept_sgd + slope_sgd * education_years
    mse_sgd = np.mean((income_values - predictions_sgd) ** 2)
    
    # calculando tambem a soluçao analitica para comparaçao
    print("\n___SOLUÇAO ANALITICA___")
    features_with_bias = np.c_[np.ones((len(features), 1)), features]
    theta_analytical = np.linalg.inv(features_with_bias.T @ features_with_bias) @ features_with_bias.T @ target
    intercept_analytical, slope_analytical = theta_analytical[0], theta_analytical[1]
    print(f"Theta (intercept): {intercept_analytical:.4f}")
    print(f"Theta (slope): {slope_analytical:.4f}")
    
    # calculando MSE da soluçao analitica
    predictions_analytical = intercept_analytical + slope_analytical * education_years
    mse_analytical = np.mean((income_values - predictions_analytical) ** 2)
    print(f"MSE: {mse_analytical:.4f}")
    
    # comparando entre os metodos
    print("\n___COMPARAÇAO SGD vs ANALITICA___")
    print(f"Diferença intercept: {abs(intercept_sgd - intercept_analytical):.4f}")
    print(f"Diferença slope: {abs(slope_sgd - slope_analytical):.4f}")
    print(f"Diferença MSE: {abs(mse_sgd - mse_analytical):.4f}")
    
    # criando visualizaçoes
    plt.figure(figsize=(15, 5))
    
    # criando grafico 1: RENDA x ANOS DE ESTUDO (comparacao SGD x Analitica)
    plt.subplot(1, 3, 1)
    
    education_range = np.array([education_years.min(), education_years.max()])
    income_predicted_sgd = intercept_sgd + slope_sgd * education_range
    income_predicted_analytical = intercept_analytical + slope_analytical * education_range
    
    plt.scatter(education_years, income_values, color="blue", alpha=0.5, label="Dados")
    plt.plot(education_range, income_predicted_sgd, "r-", linewidth=2, label=f"SGD (w1={slope_sgd:.2f})")
    plt.plot(education_range, income_predicted_analytical, "g--", linewidth=2, 
             label=f"Analítica (w1={slope_analytical:.2f})")
    plt.xlabel("Anos de Estudo")
    plt.ylabel("Renda")
    plt.title("Renda x Anos de Estudo\n(SGD vs Solução Analítica)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # criando grafico 2: CURVA DE CONVERGENCIA (MSE ao longo das epocas)
    plt.subplot(1, 3, 2)
    
    epochs_plot = list(range(0, EPOCHS, 10)) + [EPOCHS]
    plt.plot(epochs_plot[:len(model_sgd.cost_history)], model_sgd.cost_history, 'b-', linewidth=2)
    plt.axhline(y=mse_analytical, color='g', linestyle='--', 
                label=f'MSE Analítico: {mse_analytical:.2f}')
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Convergência do SGD\n(MSE ao longo do treinamento)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # criando grafico 3: MSE x COEFICIENTE ANGULAR
    plt.subplot(1, 3, 3)
    
    def calculate_mse(test_slope):
        predicted_income = intercept_sgd + test_slope * education_years
        return np.mean((income_values - predicted_income) ** 2)
    
    slope_range = np.linspace(slope_sgd - 2, slope_sgd + 2, 100)
    mse_values = [calculate_mse(s) for s in slope_range]
    
    plt.plot(slope_range, mse_values, "b-", linewidth=2)
    plt.axvline(x=slope_sgd, color="red", linestyle="-", linewidth=2,
                label=f"SGD w1 = {slope_sgd:.2f}")
    plt.axvline(x=slope_analytical, color="green", linestyle="--", linewidth=2,
                label=f"Analítico w1 = {slope_analytical:.2f}")
    plt.xlabel("Coeficiente Angular (w1)")
    plt.ylabel("MSE")
    plt.title("MSE x Coeficiente Angular")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n___ANALISE COMPLETA___")
    print(f"O SGD convergiu proximo a soluçao analitica")
    print(f"MSE SGD: {mse_sgd:.4f}")
    print(f"MSE Analitico: {mse_analytical:.4f}")


if __name__ == "__main__":
    main()
