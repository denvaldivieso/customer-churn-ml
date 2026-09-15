from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Columnas numéricas del dataset. TotalCharges es la que tiene los
# 26 nulos que detectamos en el EDA.
COLUMNAS_NUMERICAS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Columnas categóricas (de texto). Son todas las columnas de servicios
# y datos del cliente, sin contar customerID (que no se usa como
# feature) ni Churn (que es el target, no una entrada del modelo).
COLUMNAS_CATEGORICAS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def construir_preprocesador():
    """
    Arma y devuelve el ColumnTransformer que se va a aplicar a los datos,
    tanto para entrenar como para predecir más adelante.
    """

    # Para las columnas numéricas: primero rellenamos los nulos con la
    # mediana (es una medida robusta, no la afectan valores extremos),
    # y después escalamos, para que todas las columnas numéricas queden
    # en rangos comparables entre sí (esto ayuda especialmente al modelo
    # de regresión logística).
    transformador_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Para las columnas categóricas: si hay nuelos los rellenamos
    # con el valor más frecuente de esa columna, y después convertimos
    # cada categoría en columnas de 0 y 1 con OneHotEncoder, porque los
    # modelos de scikit-learn no pueden trabajar directamente con texto.
    transformador_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combinamos ambos transformadores en un solo objeto, indicando a
    # qué columnas se le aplica cada uno.
    preprocesador = ColumnTransformer(transformers=[
        ("num", transformador_numerico, COLUMNAS_NUMERICAS),
        ("cat", transformador_categorico, COLUMNAS_CATEGORICAS),
    ])

    return preprocesador