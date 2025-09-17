import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# carregando os dados do arquivo cvs
dataset = pd.read_csv("Income1.csv", index_col=0)

educationYears = dataset["Education"].values
incomeValues = dataset["Income"].values

plt.figure(figsize=(10, 5))

# matriz estendida
xMatrix = np.c_[np.ones(len(educationYears)), educationYears]

# regressao linear usando os minimos quadrados
theta = np.linalg.inv(xMatrix.T @ xMatrix) @ xMatrix.T @ incomeValues
intercept, slope = theta[0], theta[1]

# grafico de RENDA x ANOS DE ESTUDO
xInterval = np.array([educationYears.min(), educationYears.max()])
yPredicted = intercept + slope * xInterval

plt.subplot(1, 2, 1)
plt.scatter(educationYears, incomeValues, color="blue", alpha=0.5, label="Dados")

plt.plot(xInterval, yPredicted, "r-", label="Regressão Linear")
plt.xlabel("Anos de Estudo")
plt.ylabel("Renda")
plt.title("Renda x Anos de Estudo")
plt.legend()

# grafico do MSE x COEFICIENTE ANGULAR
plt.subplot(1, 2, 2)

def calcMSE(test_slope):
    predictedIncome = intercept + test_slope * educationYears
    return np.mean((incomeValues - predictedIncome) ** 2)

slopeRange = np.linspace(slope - 2, slope + 2, 100)
mseValues = [calcMSE(s) for s in slopeRange]

# MSE otimo
mseOptimal = calcMSE(slope)

plt.plot(slopeRange, mseValues, "b-")
plt.axvline(x=slope, color="red", linestyle="--", 
            label= f"w1 = {slope:.2f}")
plt.xlabel("Coeficiente Angular")
plt.ylabel("MSE")
plt.title("MSE x Coeficiente Angular")
plt.legend()

plt.tight_layout()
plt.show()
