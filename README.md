# SALUD-IA 🏥🤖

**Anticipar para planificar mejor.**

SALUD-IA es un prototipo de inteligencia artificial orientado a apoyar la planificación de recursos en servicios de salud mediante el análisis de datos históricos.

El sistema integra dos modelos de aprendizaje automático para anticipar la demanda de atención e identificar patrones asociados con el riesgo de inasistencia a citas. A partir de estas estimaciones, presenta indicadores que pueden apoyar la toma de decisiones operativas.

🌐 **Prototipo en vivo:** https://salud-ia.streamlit.app

## Funcionalidades

### 📈 Predicción de demanda
Estima la demanda del siguiente día utilizando patrones históricos de llegadas y un modelo Random Forest.

### 👤 Riesgo de inasistencia
Genera un puntaje estimado de riesgo de inasistencia a partir de variables relacionadas con la cita y antecedentes de asistencia.

### 🏥 Planificación de recursos
Compara la demanda prevista con la capacidad disponible ingresada por el personal y muestra indicadores de utilización y posibles escenarios de saturación.

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- Altair
- joblib

## Fuentes de datos

SALUD-IA utiliza dos conjuntos de datos históricos reales y de acceso público para desarrollar y evaluar los modelos de aprendizaje automático del prototipo.

### 1. Riesgo de inasistencia

**Dataset:** Medical Appointment No Shows  
**Autor:** Joni Hoppen / Aquarela Analytics  
**Plataforma:** Kaggle  
**Archivo utilizado:** `KaggleV2-May-2016.csv`  
**Registros originales:** 110 527 citas médicas  
**País:** Brasil  

**Fuente:**  
https://www.kaggle.com/datasets/joniarroba/noshowappointments

Este conjunto de datos contiene información sobre citas médicas y si los pacientes asistieron o no. En SALUD-IA se utiliza para entrenar un modelo Random Forest que identifica patrones asociados con el riesgo de inasistencia.

### 2. Predicción de demanda

**Estudio:** *Forecasting daily emergency department arrivals using high-dimensional multivariate data: a feature selection approach*  

**Autores:** Jalmari Tuominen, Francesco Lomio, Niku Oksala, Ari Palomäki, Jaakko Peltonen, Heikki Huttunen y Antti Roine  
**Año:** 2022  
**Revista:** BMC Medical Informatics and Decision Making  
**Institución:** Tampere University Hospital, Finlandia  
**Periodo de los datos:** 1 de junio de 2015 al 19 de junio de 2019  
**Archivo utilizado:** `12911_2022_1878_MOESM1_ESM.xlsx`

**Artículo original:**  
https://doi.org/10.1186/s12911-022-01878-7

**Conjunto de datos:**  
https://doi.org/10.6084/m9.figshare.19785656.v1

SALUD-IA utiliza los registros históricos de llegadas al servicio de emergencias para entrenar un modelo Random Forest que estima la demanda del siguiente día a partir de patrones históricos.

El modelo desarrollado para SALUD-IA es independiente de los modelos evaluados en la investigación original.

### Limitaciones de los datos

Aunque los conjuntos de datos utilizados contienen información real, ninguno corresponde a datos actuales de instituciones de salud de Costa Rica.

Por esta razón, SALUD-IA debe considerarse una **prueba de concepto** y sus resultados no representan modelos validados para la población costarricense ni para la CCSS.

Una implementación real requeriría datos institucionales actualizados, validación local, evaluación del desempeño de los modelos y reentrenamiento periódico antes de utilizar el sistema como apoyo operativo.

## Uso responsable

SALUD-IA está diseñado como una herramienta de **apoyo a la planificación**, no como un sistema de decisión autónoma.

El prototipo:

- no realiza diagnósticos;
- no determina prioridades clínicas;
- no cancela citas automáticamente;
- no debe utilizarse para limitar el acceso de una persona a servicios de salud;
- requiere revisión humana de sus resultados.

Las predicciones reflejan patrones encontrados en datos históricos y no representan certezas sobre el comportamiento individual de una persona.

## Ejecutar el prototipo

Instalar las dependencias:

```bash
pip install -r requirements.txt