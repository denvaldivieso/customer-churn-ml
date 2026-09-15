# Customer Churn - Laboratorio de Minería de Datos

Proyecto integrador de la materia Laboratorio de Minería de Datos (ISTEA).
El objetivo es construir un sistema que prediga si un cliente de una
empresa de telecomunicaciones va a abandonar el servicio (churn) o no,
partiendo de sus datos históricos.

Esta primera entrega cubre la etapa de exploración, armado del pipeline
de preprocesamiento, entrenamiento y comparación de modelos, el
registro de experimentos y del modelo candidato.

## 1. El problema y los datos

El dataset fue provisto y contiene 7.043 clientes
históricos, con variables como: tipo de contrato, forma de pago,
antigüedad y servicios contratados, además de la variable que queremos
predecir (Churn: Yes/No).

Encontramos que la variable TotalCharges tiene 26 valores faltantes y
que el target está desbalanceado: un 73,6% de los clientes no se van y
un 26,4% sí. Por este desbalance, no usamos Accuracy como métrica
principal para elegir el modelo, sino que priorizamos Recall y F1,
porque nos interesa más detectar la mayor cantidad posible de clientes
que realmente se van (un falso negativo significa perder a ese cliente
sin intentar retenerlo).

El detalle completo del análisis está en `notebooks/01_eda.ipynb`.

## 2. Estructura del proyecto
customer-churn-ml/
├── data/
│ ├── raw/ # dataset histórico (versionado con DVC)
│ ├── production/ # datos para la etapa de monitoreo (entrega final)
│ ├── scoring/ # datos de prueba para la API (entrega 2)
│ └── metadata/ # diccionario de datos y schema
├── notebooks/
│ ├── 01_eda.ipynb # análisis exploratorio
│ └── 02_train_test_split.ipynb # partición train/test
├── src/
│ ├── features/
│ │ └── build_pipeline.py # pipeline de preprocesamiento
│ └── training/
│ └── train.py # script de entrenamiento
├── requirements.txt
└── README.md

## 3. Cómo instalar el proyecto

Con Python 3.11 instalado, desde la raíz del proyecto: pip install -r requirements.txt

## 4. Cómo recuperar los datos

El dataset histórico no está en este repositorio de forma directa, está
versionado con DVC y guardado en DagsHub. Para bajarlo: dvc pull

Esto trae el archivo `data/raw/customer_churn_historical.csv` a la
carpeta local. El repositorio en DagsHub es:
https://dagshub.com/denvaldivieso/customer-churn-ml

## 5. Cómo entrenar los modelos

El entrenamiento se corre desde la terminal, sin depender de ningún
notebook. El script permite elegir qué modelo entrenar y con qué
hiperparámetros:
python -m src.training.train --model baseline
python -m src.training.train --model logistic --C 1.0
python -m src.training.train --model random_forest --n_estimators 200 --max_depth 10

Cada ejecución hace lo siguiente: carga el dataset histórico, separa un
80% para entrenamiento y un 20% para test (de forma reproducible, con un
random_state fijo y manteniendo la proporción del target en ambas
partes), aplica el pipeline de preprocesamiento, entrena el modelo
elegido, calcula las métricas sobre el conjunto de test, y registra
todo (parámetros, métricas y el modelo entrenado) en MLflow, en el
proyecto conectado a DagsHub.

## 6. Preprocesamiento

El pipeline de preprocesamiento (`src/features/build_pipeline.py`)
resuelve dos cosas que encontramos en el análisis exploratorio: los 26
valores faltantes de TotalCharges los completa con la mediana y la
conversión de las columnas categóricas a un formato que los modelos
puedan usar, a través de OneHotEncoder. Este mismo pipeline se usa siempre,
tanto para entrenar como para cualquier predicción futura, así nos
asegura que a los datos se les aplican siempre las mismas
transformaciones.

## 7. Modelos entrenados y resultados

Entrenamos y comparamos 6 combinaciones distintas, todas registradas
como experimentos en MLflow:

| Modelo | Hiperparámetros | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline (DummyClassifier) | - | 0.736 | 0.00 | 0.00 | 0.00 | 0.50 |
| Regresión logística | C = 1.0 | 0.794 | 0.66 | 0.45 | **0.54** | **0.81** |
| Regresión logística | C = 0.1 | 0.791 | 0.66 | 0.44 | 0.53 | 0.81 |
| Random Forest | valores por defecto | 0.783 | 0.64 | 0.40 | 0.49 | 0.79 |
| Random Forest | n_estimators=200, max_depth=10 | 0.791 | 0.67 | 0.41 | 0.51 | 0.80 |
| Random Forest | n_estimators=300, max_depth=6 | 0.782 | 0.69 | 0.31 | 0.43 | 0.80 |

El modelo elegido como candidato es la **regresión logística con C=1.0**,
porque es el que mejor equilibra Recall y F1 entre las 6 combinaciones
probadas, sin sacrificar demasiado la Precision. Ningún ajuste de
hiperparámetros del Random Forest logró superarla en las métricas que
más nos interesan para este problema.

Este modelo quedó registrado en el Model Registry de MLflow bajo el
nombre `churn_logistic_regression`.

## 8. Trazabilidad

Todos los experimentos incluyendo los que dieron peor resultado, como
el Random Forest con distintas configuraciones, quedaron conservados en
MLflow. No se borro ninguno, para de esta fomra poder consultar en cualquier momento
qué se probó, con qué datos, y qué métricas dio. El proyecto de MLflow
está conectado al mismo repositorio de DagsHub mencionado en el punto 4.