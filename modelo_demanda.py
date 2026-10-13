import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. CARGAR DATOS
# --------------------------------------------------

archivo = "data/12911_2022_1878_MOESM1_ESM.xlsx"

datos = pd.read_excel(
    archivo,
    skiprows=3,
    names=["Date", "Arrivals", "Weekday"]
)

datos["Date"] = pd.to_datetime(datos["Date"])
datos["Arrivals"] = pd.to_numeric(datos["Arrivals"])

datos = datos.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------
# 2. CREAR VARIABLES TEMPORALES
# --------------------------------------------------

datos["DayOfWeek"] = datos["Date"].dt.dayofweek
datos["Month"] = datos["Date"].dt.month

# Demanda observada anteriormente
datos["Lag1"] = datos["Arrivals"].shift(1)
datos["Lag7"] = datos["Arrivals"].shift(7)

# IMPORTANTE:
# shift(1) evita utilizar la demanda del propio día
datos["Rolling7"] = (
    datos["Arrivals"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

datos["Rolling28"] = (
    datos["Arrivals"]
    .shift(1)
    .rolling(window=28)
    .mean()
)

# Las primeras filas no tienen suficiente historial
datos_modelo = datos.dropna().copy()


# --------------------------------------------------
# 3. VARIABLES DEL MODELO
# --------------------------------------------------

variables = [
    "DayOfWeek",
    "Month",
    "Lag1",
    "Lag7",
    "Rolling7",
    "Rolling28"
]

X = datos_modelo[variables]
y = datos_modelo["Arrivals"]


# --------------------------------------------------
# 4. DIVISIÓN CRONOLÓGICA 80/20
# --------------------------------------------------

punto_corte = int(len(datos_modelo) * 0.80)

X_train = X.iloc[:punto_corte]
X_test = X.iloc[punto_corte:]

y_train = y.iloc[:punto_corte]
y_test = y.iloc[punto_corte:]

fechas_test = datos_modelo["Date"].iloc[punto_corte:]


# --------------------------------------------------
# 5. ENTRENAR RANDOM FOREST
# --------------------------------------------------

modelo = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

modelo.fit(X_train, y_train)


# --------------------------------------------------
# 6. REALIZAR PREDICCIONES
# --------------------------------------------------

predicciones = modelo.predict(X_test)


# --------------------------------------------------
# 7. EVALUAR
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predicciones
)

rmse = mean_squared_error(
    y_test,
    predicciones
) ** 0.5


print("MODELO DE PREDICCIÓN DE DEMANDA")
print("--------------------------------")

print("\nRegistros utilizados:", len(datos_modelo))

print("\nDIVISIÓN CRONOLÓGICA:")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

print(
    "Entrenamiento desde:",
    datos_modelo.iloc[0]["Date"]
)

print(
    "Entrenamiento hasta:",
    datos_modelo.iloc[punto_corte - 1]["Date"]
)

print(
    "Prueba desde:",
    datos_modelo.iloc[punto_corte]["Date"]
)

print(
    "Prueba hasta:",
    datos_modelo.iloc[-1]["Date"]
)

print("\nRESULTADOS RANDOM FOREST:")
print("MAE:", mae)
print("RMSE:", rmse)


# --------------------------------------------------
# 8. COMPARAR REAL VS PREDICHO
# --------------------------------------------------

resultados = pd.DataFrame({
    "Date": fechas_test.values,
    "Real": y_test.values,
    "Predicted": predicciones
})

print("\nÚLTIMAS PREDICCIONES:")
print(resultados.tail(10))

# --------------------------------------------------
# 9. COMPARAR CONTRA UN MODELO BASE
# --------------------------------------------------

# Predicción simple:
# asumir que hoy tendrá la misma demanda
# que el mismo día de la semana anterior
predicciones_base = X_test["Lag7"]

mae_base = mean_absolute_error(
    y_test,
    predicciones_base
)

rmse_base = mean_squared_error(
    y_test,
    predicciones_base
) ** 0.5

print("\nRESULTADOS MODELO BASE (LAG 7):")
print("MAE:", mae_base)
print("RMSE:", rmse_base)

mejora_mae = (
    (mae_base - mae)
    / mae_base
) * 100

print("\nMEJORA DEL RANDOM FOREST:")
print("Reducción del MAE:", mejora_mae, "%")

# --------------------------------------------------
# GUARDAR MODELO FINAL
# --------------------------------------------------

joblib.dump(
    modelo,
    "models/modelo_demanda.pkl"
)

print("\nModelo de demanda guardado correctamente.")