import pandas as pd

datos = pd.read_csv("data/KaggleV2-May-2016.csv")

print(datos.head())

print("\nCOLUMNAS:")
print(datos.columns.tolist())

print("\nTAMAÑO DEL DATASET:")
print(datos.shape)

print("\nTIPOS DE DATOS:")
print(datos.dtypes)

print("\nDISTRIBUCIÓN DE NO-SHOW:")
print(datos["No-show"].value_counts())

print("\nVALORES FALTANTES:")
print(datos.isnull().sum())

print("\nRESUMEN DE EDADES:")
print(datos["Age"].describe())

print("\nVALORES ÚNICOS DE VARIABLES BINARIAS:")
for columna in ["Gender", "Scholarship", "Hipertension",
                "Diabetes", "Alcoholism", "Handcap",
                "SMS_received", "No-show"]:
    print(f"\n{columna}:")
    print(datos[columna].value_counts())

datos["ScheduledDay"] = pd.to_datetime(datos["ScheduledDay"])
datos["AppointmentDay"] = pd.to_datetime(datos["AppointmentDay"])

datos["WaitingDays"] = (
    datos["AppointmentDay"].dt.normalize()
    - datos["ScheduledDay"].dt.normalize()
).dt.days

print("\nRESUMEN DE DÍAS DE ESPERA:")
print(datos["WaitingDays"].describe())

print("\nCITAS CON ESPERA NEGATIVA:")
print(datos[datos["WaitingDays"] < 0][
    ["ScheduledDay", "AppointmentDay", "WaitingDays"]
])
