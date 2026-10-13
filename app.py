import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

st.set_page_config(
    page_title="SALUD-IA",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# CARGAR MODELOS Y DATOS
# --------------------------------------------------

modelo_demanda = joblib.load(
    "models/modelo_demanda.pkl"
)

modelo_inasistencia = joblib.load(
    "models/modelo_inasistencia.pkl"
)

archivo_demanda = "data/12911_2022_1878_MOESM1_ESM.xlsx"

datos_demanda = pd.read_excel(
    archivo_demanda,
    skiprows=3,
    names=["Date", "Arrivals", "Weekday"]
)

datos_demanda["Date"] = pd.to_datetime(
    datos_demanda["Date"]
)

datos_demanda["Arrivals"] = pd.to_numeric(
    datos_demanda["Arrivals"]
)

datos_demanda = datos_demanda.sort_values(
    "Date"
).reset_index(drop=True)

# --------------------------------------------------
# ENCABEZADO
# --------------------------------------------------

st.markdown("""
<div style="padding: 28px 32px; border-radius: 18px; background: linear-gradient(135deg, #14213d, #1f4068);">
<p style="font-size:14px; font-weight:600; letter-spacing:2px; opacity:0.8; margin:0;">
INTELIGENCIA ARTIFICIAL · GESTIÓN EN SALUD
</p>
<h1 style="font-size:42px; margin:8px 0 0 0;">SALUD-IA</h1>
<h3 style="margin:6px 0 12px 0;">Anticipar para planificar mejor.</h3>
<p style="font-size:15px; opacity:0.85; margin:0; max-width:850px;">
Sistema inteligente para anticipar la demanda, identificar patrones de riesgo de inasistencia y apoyar la planificación de recursos en servicios de salud.
</p>
</div>
<br>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MÓDULOS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📈 Predicción de demanda",
    "👤 Riesgo de inasistencia",
    "🏥 Planificación"
])

with tab1:
    st.header("Predicción de demanda")

    st.info(
    "🧪 Demostración con datos históricos abiertos (2015–2019). "
    "La fecha mostrada corresponde al día siguiente al último registro "
    "disponible. En una implementación real, SALUD-IA utilizaría datos "
    "actualizados de la institución y requeriría reentrenamiento y "
    "validación antes de su uso operativo."
    )

    st.write(
        "Estimación de la demanda del siguiente día "
        "a partir del comportamiento histórico."
    )

    ultima_fecha = datos_demanda["Date"].max()
    fecha_prediccion = ultima_fecha + pd.Timedelta(days=1)

    # Información histórica necesaria
    lag1 = datos_demanda.iloc[-1]["Arrivals"]
    lag7 = datos_demanda.iloc[-7]["Arrivals"]

    rolling7 = (
        datos_demanda["Arrivals"]
        .tail(7)
        .mean()
    )

    rolling28 = (
        datos_demanda["Arrivals"]
        .tail(28)
        .mean()
    )

    entrada = pd.DataFrame({
        "DayOfWeek": [fecha_prediccion.dayofweek],
        "Month": [fecha_prediccion.month],
        "Lag1": [lag1],
        "Lag7": [lag7],
        "Rolling7": [rolling7],
        "Rolling28": [rolling28]
    })

    demanda_predicha = modelo_demanda.predict(
        entrada
    )[0]

    demanda_predicha = round(demanda_predicha)

    diferencia_promedio = demanda_predicha - rolling7
    porcentaje_cambio = (
        diferencia_promedio / rolling7
    ) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fecha simulada",
            fecha_prediccion.strftime("%d/%m/%Y")
        )

    with col2:
        st.metric(
            "Demanda prevista",
            f"{demanda_predicha} llegadas",
            delta=f"{porcentaje_cambio:+.1f}% vs. promedio reciente",
            delta_color="off"
        )

    with col3:
        st.metric(
            "Promedio últimos 7 días",
            f"{rolling7:.0f} llegadas"
        )

    st.subheader("Demanda reciente")

    historico = (
    datos_demanda[
        ["Date", "Arrivals"]
    ]
    .tail(30)
    .copy()
)

    historico["Histórico"] = historico["Arrivals"]
    historico["Predicción"] = None

    ultima_fecha_real = datos_demanda.iloc[-1]["Date"]
    ultima_demanda_real = datos_demanda.iloc[-1]["Arrivals"]

    fila_inicio_prediccion = pd.DataFrame({
        "Date": [ultima_fecha_real],
        "Arrivals": [None],
        "Histórico": [None],
        "Predicción": [ultima_demanda_real]
    })

    fila_prediccion = pd.DataFrame({
        "Date": [fecha_prediccion],
        "Arrivals": [None],
        "Histórico": [None],
        "Predicción": [demanda_predicha]
    })

    grafico = pd.concat(
        [
            historico,
            fila_inicio_prediccion,
            fila_prediccion
        ],
        ignore_index=True
    )

    grafico = grafico[
        ["Date", "Histórico", "Predicción"]
    ].set_index("Date")

    st.line_chart(grafico)

    st.caption(
        "Predicción generada mediante Random Forest "
        "a partir de patrones históricos de demanda."
    )

with tab2:
    st.header("Riesgo de inasistencia")

    st.info(
        "🧪 Demostración basada en datos históricos abiertos. "
        "El modelo no está validado para la población costarricense "
        "y sus resultados representan únicamente una prueba de concepto."
    )

    st.write(
        "Ingrese información de una cita para estimar su "
        "nivel de riesgo de inasistencia."
    )

    col1, col2 = st.columns(2)

    with col1:
        edad = st.number_input(
            "Edad",
            min_value=0,
            max_value=115,
            value=35
        )

        dias_espera = st.number_input(
            "Días entre la programación y la cita",
            min_value=0,
            max_value=180,
            value=7
        )

        dia_cita = st.selectbox(
            "Día de la cita",
            [
                "Lunes",
                "Martes",
                "Miércoles",
                "Jueves",
                "Viernes",
                "Sábado",
                "Domingo"
            ]
        )

        dia_programacion = st.selectbox(
            "Día en que se programó",
            [
                "Lunes",
                "Martes",
                "Miércoles",
                "Jueves",
                "Viernes",
                "Sábado",
                "Domingo"
            ]
        )

    with col2:
        sms = st.selectbox(
            "¿Recibió SMS?",
            ["Sí", "No"]
        )

        citas_previas = st.number_input(
            "Citas anteriores conocidas",
            min_value=0,
            value=0
        )

        faltas_previas = st.number_input(
            "Inasistencias anteriores",
            min_value=0,
            value=0
        )

    dias = {
        "Lunes": 0,
        "Martes": 1,
        "Miércoles": 2,
        "Jueves": 3,
        "Viernes": 4,
        "Sábado": 5,
        "Domingo": 6
    }

    if citas_previas > 0:
        tasa_faltas = faltas_previas / citas_previas
    else:
        tasa_faltas = 0.0

    if faltas_previas > citas_previas:
        st.warning(
            "Las inasistencias anteriores no pueden superar "
            "el número de citas anteriores."
        )

    analizar = st.button(
        "Analizar riesgo de inasistencia",
        type="primary"
    )

    if analizar and faltas_previas <= citas_previas:

        entrada_inasistencia = pd.DataFrame({
            "Age": [edad],
            "WaitingDays": [dias_espera],
            "AppointmentWeekday": [dias[dia_cita]],
            "ScheduledWeekday": [dias[dia_programacion]],
            "SMS_received": [1 if sms == "Sí" else 0],
            "PreviousAppointments": [citas_previas],
            "PreviousNoShows": [faltas_previas],
            "PreviousNoShowRate": [tasa_faltas]
        })

        probabilidad = modelo_inasistencia.predict_proba(
            entrada_inasistencia
        )[0][1]

        porcentaje = probabilidad * 100

        st.divider()

        st.subheader("Resultado")

        st.metric(
            "Puntaje estimado de riesgo de inasistencia",
            f"{porcentaje:.1f}%"
        )

        st.progress(
            min(max(probabilidad, 0.0), 1.0)
        )

        st.caption(
            "El puntaje refleja patrones identificados por el modelo "
            "en los datos históricos y no representa una certeza "
            "sobre la conducta individual de una persona."
        )

        if probabilidad < 0.30:
            st.success(
                "Riesgo estimado bajo."
            )

        elif probabilidad < 0.60:
            st.warning(
                "Riesgo estimado moderado. "
                "Puede considerarse reforzar la confirmación "
                "o el recordatorio de la cita."
            )

        else:
            st.error(
                "Riesgo estimado alto. "
                "Puede considerarse un refuerzo adicional "
                "de confirmación o recordatorio."
            )

        st.caption(
            "Esta estimación es únicamente una herramienta de apoyo. "
            "No debe utilizarse para cancelar citas, limitar el acceso "
            "a servicios ni tomar decisiones clínicas."
        )

with tab3:
    st.header("Planificación de recursos")

    st.write(
        "Compare la demanda prevista por SALUD-IA con la "
        "capacidad disponible para apoyar la planificación."
    )

    st.info(
        "La capacidad es ingresada por el personal. "
        "SALUD-IA genera indicadores de apoyo, pero la decisión "
        "final permanece bajo revisión humana."
    )

    capacidad = st.number_input(
        "Capacidad disponible para el día",
        min_value=1,
        value=250,
        step=1
    )

    utilizacion = (
        demanda_predicha / capacidad
    ) * 100

    diferencia = capacidad - demanda_predicha

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Demanda prevista",
            f"{demanda_predicha} llegadas"
        )

    with col2:
        st.metric(
            "Capacidad disponible",
            f"{capacidad} atenciones"
        )

    with col3:
        st.metric(
            "Utilización prevista",
            f"{utilizacion:.1f}%"
        )

    st.write("**Uso previsto de la capacidad**")

    st.progress(
        min(utilizacion / 100, 1.0)
    )

    st.subheader("Evaluación de capacidad")

    if utilizacion > 100:
        st.error(
            "⚠️ Riesgo de saturación: la demanda prevista "
            "supera la capacidad disponible."
        )

        st.write(
            f"Se estiman aproximadamente "
            f"{abs(diferencia)} llegadas por encima "
            f"de la capacidad indicada."
        )

        st.write(
            "**Apoyo a la planificación:** revisar disponibilidad "
            "de personal, espacios y otros recursos antes de tomar "
            "cualquier decisión operativa."
        )

    elif utilizacion >= 90:
        st.warning(
            "🟡 Capacidad cercana al límite: la demanda prevista "
            "utilizaría al menos el 90% de la capacidad disponible."
        )

        st.write(
            f"Margen estimado disponible: "
            f"{diferencia} atenciones."
        )

        st.write(
            "**Apoyo a la planificación:** mantener seguimiento "
            "de la demanda y revisar posibles ajustes preventivos."
        )

    else:
        st.success(
            "🟢 Capacidad dentro del margen previsto."
        )

        st.write(
            f"Margen estimado disponible: "
            f"{diferencia} atenciones."
        )

    st.caption(
        "Los umbrales utilizados para visualizar el nivel de "
        "capacidad son demostrativos y no representan criterios "
        "clínicos ni protocolos institucionales."
    )

st.divider()

st.caption(
    "SALUD-IA es un prototipo de apoyo operativo. "
    "No realiza diagnósticos, priorización clínica ni "
    "decisiones automáticas sobre pacientes. "
    "Los resultados requieren revisión humana."
)