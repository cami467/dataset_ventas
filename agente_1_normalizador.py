#Instalaciones dependencias
!pip install pandas scikit-learn tabulate

# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabulate import tabulate
print(" Librerías cargadas correctamente")

#Cargar el dataset
import pandas as pd

# Si ya está en la carpeta de trabajo
df = pd.read_csv("/content/Telco-Customer-Churn.csv")
print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
print("\nColumnas:")
print(df.columns.tolist())
df.head(3)

# Revisión inicial del dataset
print("=== TIPOS DE DATOS ===")
print(df.dtypes)
print("\n=== ESTADÍSTICAS ===")
df.describe()

# Detección de nulos y vacíos
# TotalCharges tiene espacios vacíos que Pandas NO detecta como NaN
df['TotalCharges'] = df['TotalCharges'].str.strip()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Ahora sí aparecen como NaN
nulos = df.isnull().sum()
df_nulos = pd.DataFrame({
    'Columna': nulos.index,
    'Nulos': nulos.values,
    'Porcentaje (%)': (nulos.values / len(df) * 100).round(2)
})
df_nulos = df_nulos[df_nulos['Nulos'] > 0]
print(tabulate(df_nulos, headers='keys', tablefmt='psql', showindex=False))

# Imputar nulos
#Imputar TotalCharges con la mediana
mediana = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(mediana)
print(f"TotalCharges imputado con mediana: {mediana}")
print(f"Nulos restantes: {df['TotalCharges'].isnull().sum()}")

#-------------------------------------------
# Eliminar columna irrelevante
# -----------------------------------------
# Eliminar customerID (no aporta al modelo)
df = df.drop(columns=['customerID'])
print(f" customerID eliminado. Shape actual: {df.shape}")


#-------------------------------------------
# Encoding de columnas
# -----------------------------------------
# Encoding
cols_binarias = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                 'Churn', 'gender']

for col in cols_binarias:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})

# Columnas con múltiples categorías → LabelEncoder
cols_multi = ['MultipleLines', 'InternetService', 'OnlineSecurity',
              'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

le = LabelEncoder()
for col in cols_multi:
    df[col] = le.fit_transform(df[col])

print("Encoding aplicado")
print(df.dtypes)

#-------------------------------------------
# ESCALADO DE DATOS NUMÉRICOS
# -----------------------------------------

# Escalado (StandardScaler) sobre columnas numéricas continuas
# Se aplica DESPUÉS del encoding
cols_escalar = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = StandardScaler()
df[cols_escalar] = scaler.fit_transform(df[cols_escalar])

print(" Escalado aplicado a:", cols_escalar)
df[cols_escalar].describe().round(3)

# --------------------------------------
# Guardar y descargar Dataset limpio
# --------------------------------------

from google.colab import files

df.to_csv("telco_limpio.csv", index=False)
print(f"Dataset limpio guardado: {df.shape[0]} filas x {df.shape[1]} columnas")
files.download("telco_limpio.csv")