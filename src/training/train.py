"""
Script de entrenamiento para el proyecto de Customer Churn.

Este archivo entrena un modelo (baseline, regresión logística o
random forest), evalúa sus métricas sobre el conjunto de test, y
registra todo el experimento en MLflow (params, métricas y el
modelo entrenado como artefacto).

Se ejecuta desde la terminal, por ejemplo:
    python -m src.training.train --model baseline
    python -m src.training.train --model logistic --C 1.0
    python -m src.training.train --model random_forest --n_estimators 200 --max_depth 10
"""

import argparse

import dagshub
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.build_pipeline import construir_preprocesador

# Ruta al dataset histórico. Se usa siempre este archivo para entrenar,
# nunca el de "production" (ese se reserva para la etapa de monitoreo).
RUTA_DATASET = "data/raw/customer_churn_historical.csv"


def cargar_datos():
    """
    Carga el dataset y separa las columnas de entrada (X) del target (y).
    Sacamos customerID porque es solo un identificador, no una feature.
    """
    df = pd.read_csv(RUTA_DATASET)
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    return X, y


def obtener_modelo(nombre_modelo, args):
    """
    Devuelve una instancia del modelo elegido, con los hiperparámetros
    que se pasaron por línea de comandos. Acá es donde se define cada
    uno de los 3 modelos mínimos que pide la consigna.
    """
    if nombre_modelo == "baseline":
        # Modelo "tonto": siempre predice la clase más frecuente (No).
        # Sirve como piso de comparación: si los otros modelos no le
        # ganan claramente a este, algo está mal.
        return DummyClassifier(strategy="most_frequent", random_state=42)

    if nombre_modelo == "logistic":
        # Modelo lineal. El parámetro C controla la regularización:
        # valores más chicos regularizan más (modelo más simple).
        return LogisticRegression(C=args.C, max_iter=1000, random_state=42)

    if nombre_modelo == "random_forest":
        # Modelo basado en árboles. n_estimators es la cantidad de
        # árboles, max_depth limita qué tan profundo puede crecer
        # cada uno (para evitar sobreajuste).
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )

    raise ValueError(f"Modelo desconocido: {nombre_modelo}")


def calcular_metricas(y_test, y_pred, y_proba):
    """
    Calcula todas las métricas que pide la consigna. No usamos
    Accuracy como métrica principal porque el dataset está
    desbalanceado (73,6% / 26,4%), pero la dejamos igual como dato
    de referencia.
    """
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="Yes"),
        "recall": recall_score(y_test, y_pred, pos_label="Yes"),
        "f1_score": f1_score(y_test, y_pred, pos_label="Yes"),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    # Definimos los argumentos que se pueden pasar por terminal, para
    # poder elegir el modelo y sus hiperparámetros sin tocar el código
    # cada vez que queremos probar una combinación distinta.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["baseline", "logistic", "random_forest"], required=True
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    args = parser.parse_args()

    # Conectamos MLflow al servidor de DagsHub, para que cada Run
    # quede registrado ahí y no solo en nuestra compu.
    dagshub.init(repo_owner="denvaldivieso", repo_name="customer-churn-ml", mlflow=True)

    # Cargamos los datos y hacemos la misma partición train/test que
    # ya probamos en el notebook 02: 80/20, con semilla fija y
    # estratificada por el target (por el desbalance de clases).
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Armamos el pipeline completo: preprocesamiento + modelo.
    # Esto asegura que a los datos siempre se les aplican las mismas
    # transformaciones, tanto en el entrenamiento como en una
    # predicción futura.
    preprocesador = construir_preprocesador()
    modelo = obtener_modelo(args.model, args)
    pipeline = Pipeline(steps=[
        ("preprocesador", preprocesador),
        ("modelo", modelo),
    ])

    # Arrancamos el Run de MLflow. Todo lo que pasa dentro de este
    # bloque queda asociado a ese experimento puntual.
    with mlflow.start_run(run_name=f"{args.model}"):
        # Entrenamos el pipeline completo con los datos de train.
        pipeline.fit(X_train, y_train)

        # Predecimos sobre el conjunto de test, que no se usó para
        # entrenar en ningún momento.
        y_pred = pipeline.predict(X_test)
        # Necesitamos la probabilidad de la clase "Yes" para el ROC-AUC.
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metricas = calcular_metricas(y_test, y_pred, y_proba)
        matriz_confusion = confusion_matrix(y_test, y_pred)

        # Registramos en MLflow qué modelo se entrenó y con qué
        # hiperparámetros, para poder comparar los Runs después.
        mlflow.log_param("modelo", args.model)
        if args.model == "logistic":
            mlflow.log_param("C", args.C)
        if args.model == "random_forest":
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)

        # Registramos todas las métricas calculadas.
        for nombre_metrica, valor in metricas.items():
            mlflow.log_metric(nombre_metrica, valor)

        # Guardamos el modelo ya entrenado (con el pipeline completo)
        # para poder usarlo después sin tener que entrenarlo de nuevo.
        # Le decimos que lo guarde con "cloudpickle" porque la forma
        # en la que lo intenta guardar por defecto (una librería
        # llamada skops) nos tiraba un error al encontrarse con un dato
        # de numpy que no reconocía como seguro.
        mlflow.sklearn.log_model(pipeline, artifact_path="modelo", serialization_format="cloudpickle")

        # Mostramos los resultados en la terminal, para verlos al
        # instante sin tener que ir a MLflow cada vez.
        print(f"Modelo: {args.model}")
        print("Métricas:", metricas)
        print("Matriz de confusión:")
        print(matriz_confusion)


if __name__ == "__main__":
    main()