import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==================================================
# 1. CARGAR LOS DATOS
# ==================================================

datos = pd.read_csv("data/KaggleV2-May-2016.csv")


# ==================================================
# 2. PREPARAR LAS FECHAS
# ==================================================

datos["ScheduledDay"] = pd.to_datetime(
    datos["ScheduledDay"]
)

datos["AppointmentDay"] = pd.to_datetime(
    datos["AppointmentDay"]
)

datos["WaitingDays"] = (
    datos["AppointmentDay"].dt.normalize()
    - datos["ScheduledDay"].dt.normalize()
).dt.days


# ==================================================
# 3. LIMPIAR REGISTROS IMPOSIBLES
# ==================================================

datos = datos[
    (datos["Age"] >= 0)
    & (datos["WaitingDays"] >= 0)
].copy()


# ==================================================
# 4. VARIABLE OBJETIVO
# ==================================================

# Dataset original:
# Yes = no asistió
# No  = sí asistió

# SALUD-IA:
# 1 = no asistió
# 0 = sí asistió

datos["NoShow"] = datos["No-show"].map({
    "Yes": 1,
    "No": 0
})


# ==================================================
# 5. VARIABLES DE CALENDARIO
# ==================================================

datos["AppointmentWeekday"] = (
    datos["AppointmentDay"].dt.dayofweek
)

datos["ScheduledWeekday"] = (
    datos["ScheduledDay"].dt.dayofweek
)


# ==================================================
# 6. CREAR HISTORIAL SIN VER EL FUTURO
# ==================================================

# Primero ordenamos todas las citas según el momento
# en que fueron programadas.
datos = datos.sort_values(
    ["ScheduledDay", "AppointmentDay"]
).copy()

# Aquí guardaremos el historial conocido de cada paciente.
historial = {}

previous_appointments = []
previous_no_shows = []
previous_no_show_rate = []


for fila in datos.itertuples():

    paciente = fila.PatientId
    momento_prediccion = fila.ScheduledDay

    # Si nunca hemos visto a este paciente,
    # creamos su historial vacío.
    if paciente not in historial:
        historial[paciente] = []

    # Solo usamos citas cuyo AppointmentDay ya ocurrió
    # antes del momento en que se programó esta cita.
    citas_conocidas = [
        cita
        for cita in historial[paciente]
        if cita["AppointmentDay"].date() < momento_prediccion.date()
    ]

    total_previas = len(citas_conocidas)

    faltas_previas = sum(
        cita["NoShow"]
        for cita in citas_conocidas
    )

    if total_previas > 0:
        tasa_previa = faltas_previas / total_previas
    else:
        tasa_previa = 0.0

    previous_appointments.append(total_previas)
    previous_no_shows.append(faltas_previas)
    previous_no_show_rate.append(tasa_previa)

    # Guardamos la cita actual para que pueda formar
    # parte del historial de citas futuras.
    historial[paciente].append({
        "AppointmentDay": fila.AppointmentDay,
        "NoShow": fila.NoShow
    })


datos["PreviousAppointments"] = previous_appointments
datos["PreviousNoShows"] = previous_no_shows
datos["PreviousNoShowRate"] = previous_no_show_rate


# ==================================================
# 7. ORDEN CRONOLÓGICO FINAL
# ==================================================

datos = datos.sort_values(
    "ScheduledDay"
).reset_index(drop=True)


# ==================================================
# 8. VARIABLES DEL MODELO
# ==================================================

variables = [
    "Age",
    "WaitingDays",
    "AppointmentWeekday",
    "ScheduledWeekday",
    "SMS_received",
    "PreviousAppointments",
    "PreviousNoShows",
    "PreviousNoShowRate"
]

X = datos[variables]
y = datos["NoShow"]


# ==================================================
# 9. DIVISIÓN CRONOLÓGICA 80 / 20
# ==================================================

punto_corte = int(len(datos) * 0.80)

X_train = X.iloc[:punto_corte]
X_test = X.iloc[punto_corte:]

y_train = y.iloc[:punto_corte]
y_test = y.iloc[punto_corte:]


print("Datos preparados correctamente.")
print("Registros disponibles:", len(datos))

print("\nDIVISIÓN CRONOLÓGICA:")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

print(
    "Entrenamiento desde:",
    datos.iloc[0]["ScheduledDay"]
)

print(
    "Entrenamiento hasta:",
    datos.iloc[punto_corte - 1]["ScheduledDay"]
)

print(
    "Prueba desde:",
    datos.iloc[punto_corte]["ScheduledDay"]
)

print(
    "Prueba hasta:",
    datos.iloc[-1]["ScheduledDay"]
)


# ==================================================
# 10. REVISAR EL HISTORIAL CREADO
# ==================================================

print("\nEJEMPLOS DE HISTORIAL:")

print(
    datos[
        [
            "PatientId",
            "ScheduledDay",
            "AppointmentDay",
            "PreviousAppointments",
            "PreviousNoShows",
            "PreviousNoShowRate",
            "NoShow"
        ]
    ].tail(10)
)


# ==================================================
# 11. REGRESIÓN LOGÍSTICA
# ==================================================

modelo = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

modelo.fit(
    X_train,
    y_train
)

predicciones = modelo.predict(
    X_test
)

probabilidades = modelo.predict_proba(
    X_test
)[:, 1]


print("\nRESULTADOS REGRESIÓN LOGÍSTICA:")

print(
    classification_report(
        y_test,
        predicciones
    )
)

print("\nMATRIZ DE CONFUSIÓN:")

print(
    confusion_matrix(
        y_test,
        predicciones
    )
)

print("\nROC-AUC - REGRESIÓN LOGÍSTICA:")

print(
    roc_auc_score(
        y_test,
        probabilidades
    )
)


# ==================================================
# 12. RANDOM FOREST
# ==================================================

bosque = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

bosque.fit(
    X_train,
    y_train
)

predicciones_bosque = bosque.predict(
    X_test
)

probabilidades_bosque = bosque.predict_proba(
    X_test
)[:, 1]


print("\nRESULTADOS RANDOM FOREST:")

print(
    classification_report(
        y_test,
        predicciones_bosque
    )
)

print("\nROC-AUC - RANDOM FOREST:")

print(
    roc_auc_score(
        y_test,
        probabilidades_bosque
    )
)

# --------------------------------------------------
# GUARDAR MODELO FINAL
# --------------------------------------------------

joblib.dump(
    bosque,
    "models/modelo_inasistencia.pkl",
    compress=3
)

print("\nModelo de inasistencia guardado correctamente.")