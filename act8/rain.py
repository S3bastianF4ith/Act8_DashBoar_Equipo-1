from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_FILE = Path(__file__).with_name("winequalityN.csv")

NUMERIC_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    datos = pd.read_csv(DATA_FILE)
    datos.columns = [col.strip() for col in datos.columns]
    for columna in NUMERIC_COLUMNS:
        if columna in datos.columns:
            datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
    datos["type"] = datos["type"].astype(str).str.strip()
    return datos.dropna(subset=NUMERIC_COLUMNS + ["type"]).reset_index(drop=True)


@st.cache_data
def tabla_frecuencias(serie: pd.Series, bins: int = 8) -> pd.DataFrame:
    categorias = pd.cut(serie, bins=bins, include_lowest=True)
    conteos = categorias.value_counts(sort=False)
    tabla = pd.DataFrame(
        {
            "Categoria": conteos.index.astype(str),
            "Frecuencia absoluta": conteos.astype(int).values,
        }
    )
    tabla["Frecuencia relativa"] = tabla["Frecuencia absoluta"] / tabla["Frecuencia absoluta"].sum()
    tabla["Porcentaje"] = tabla["Frecuencia relativa"] * 100
    tabla["Frecuencia acumulada"] = tabla["Frecuencia absoluta"].cumsum()
    return tabla


def estadisticas_filtradas(datos: pd.DataFrame) -> pd.DataFrame:
    resumen = datos[NUMERIC_COLUMNS].agg(["count", "mean", "median", "std", "min", "max"]).T
    resumen = resumen.rename(
        columns={
            "count": "Conteo",
            "mean": "Media",
            "median": "Mediana",
            "std": "Desviacion estandar",
            "min": "Minimo",
            "max": "Maximo",
        }
    )
    return resumen.round(3)


def grafico_tipo_vino(datos: pd.DataFrame):
    conteo = datos["type"].value_counts().sort_index()
    return px.bar(
        conteo.reset_index(),
        x="type",
        y="count",
        title="Distribucion de tipos de vino",
        labels={"type": "Tipo", "count": "Registros"},
        color_discrete_sequence=["#0f766e"],
    )


def grafico_calidad_promedio(datos: pd.DataFrame):
    resumen = datos.groupby("quality")["alcohol"].mean().reset_index()
    return px.line(
        resumen,
        x="quality",
        y="alcohol",
        markers=True,
        title="Alcohol promedio por calidad",
        labels={"quality": "Calidad", "alcohol": "Alcohol promedio"},
        color_discrete_sequence=["#1d4ed8"],
    )


def grafico_dispersion(datos: pd.DataFrame):
    return px.scatter(
        datos,
        x="residual sugar",
        y="alcohol",
        color="quality",
        title="Residual sugar vs alcohol",
        labels={"residual sugar": "Residual sugar", "alcohol": "Alcohol", "quality": "Quality"},
        opacity=0.75,
        color_continuous_scale="Viridis",
    )


def grafico_correlation(datos: pd.DataFrame):
    columnas = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "density", "pH", "sulphates", "alcohol", "quality"]
    corr = datos[columnas].corr()["quality"].drop("quality").sort_values(ascending=False)
    return px.bar(
        corr.reset_index(),
        x="index",
        y="quality",
        title="Correlacion con quality",
        labels={"index": "Variable", "quality": "Correlacion"},
        color_discrete_sequence=["#ca8a04"],
    )


def grafico_boxplot_calidad_por_tipo(datos: pd.DataFrame):
    return px.box(
        datos,
        x="type",
        y="quality",
        title="Distribucion de quality por tipo",
        labels={"type": "Tipo", "quality": "Quality"},
        color="type",
    )


def grafico_histograma_alcohol(datos: pd.DataFrame):
    return px.histogram(
        datos,
        x="alcohol",
        nbins=20,
        title="Histograma de alcohol",
        labels={"alcohol": "Alcohol", "count": "Frecuencia"},
        color_discrete_sequence=["#7c3aed"],
    )


def grafico_pH_por_tipo(datos: pd.DataFrame):
    resumen = datos.groupby("type")["pH"].mean().sort_values(ascending=False)
    return px.bar(
        resumen.reset_index(),
        x="type",
        y="pH",
        title="pH promedio por tipo",
        labels={"type": "Tipo", "pH": "pH promedio"},
        color_discrete_sequence=["#dc2626"],
    )


def _describir_grafico(titulo: str, descripcion: str):
    st.markdown(f"**{titulo}**")
    st.write(descripcion)


def _slider_rango(etiqueta: str, serie: pd.Series):
    return st.slider(
        etiqueta,
        min_value=float(serie.min()),
        max_value=float(serie.max()),
        value=(float(serie.min()), float(serie.max())),
    )


def main():
    st.set_page_config(page_title="Dashboard de Vinos", layout="wide")
    st.title("Dashboard de analisis de vinos")
    st.caption("Filtros dinamicos sobre winequalityN.csv para tabla, estadisticas y graficos.")

    datos = cargar_datos()

    with st.sidebar:
        st.header("Filtros")
        tipos_disponibles = sorted(datos["type"].dropna().unique().tolist())
        tipos = st.multiselect("Tipo de vino", tipos_disponibles, default=tipos_disponibles)

        st.subheader("Filtros por seleccion")
        calidades_disponibles = sorted(datos["quality"].dropna().unique().astype(int).tolist())
        calidades = st.multiselect("Quality", calidades_disponibles, default=calidades_disponibles)

        st.subheader("Filtros por slider")
        left_col, right_col = st.columns(2)

        with left_col:
            fixed_acidity = _slider_rango("Fixed acidity", datos["fixed acidity"])
            citric_acid = _slider_rango("Citric acid", datos["citric acid"])
            free_sulfur_dioxide = _slider_rango("Free sulfur dioxide", datos["free sulfur dioxide"])
            alcohol = _slider_rango("Alcohol", datos["alcohol"])
            pH = _slider_rango("pH", datos["pH"])
            mostrar_todo = st.checkbox("Mostrar solo 10 filas", value=True)

        with right_col:
            volatile_acidity = _slider_rango("Volatile acidity", datos["volatile acidity"])
            chlorides = _slider_rango("Chlorides", datos["chlorides"])
            total_sulfur_dioxide = _slider_rango("Total sulfur dioxide", datos["total sulfur dioxide"])
            sulphates = _slider_rango("Sulphates", datos["sulphates"])
            densidad = _slider_rango("Density", datos["density"])
            azucar = _slider_rango("Residual sugar", datos["residual sugar"])

    filtrado = datos[
        datos["type"].isin(tipos)
        & datos["quality"].isin(calidades)
        & datos["alcohol"].between(alcohol[0], alcohol[1])
        & datos["pH"].between(pH[0], pH[1])
        & datos["residual sugar"].between(azucar[0], azucar[1])
        & datos["density"].between(densidad[0], densidad[1])
        & datos["fixed acidity"].between(fixed_acidity[0], fixed_acidity[1])
        & datos["volatile acidity"].between(volatile_acidity[0], volatile_acidity[1])
        & datos["citric acid"].between(citric_acid[0], citric_acid[1])
        & datos["chlorides"].between(chlorides[0], chlorides[1])
        & datos["free sulfur dioxide"].between(free_sulfur_dioxide[0], free_sulfur_dioxide[1])
        & datos["total sulfur dioxide"].between(total_sulfur_dioxide[0], total_sulfur_dioxide[1])
        & datos["sulphates"].between(sulphates[0], sulphates[1])
    ].copy()

    total_registros = len(datos)
    filtrados = len(filtrado)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros originales", f"{total_registros:,}")
    col2.metric("Registros filtrados", f"{filtrados:,}")
    col3.metric("Tipos seleccionados", f"{len(tipos)}")
    col4.metric("Calidad media", f"{filtrado['quality'].mean():.2f}" if filtrados else "Sin datos")

    st.subheader("Tabla filtrada")
    columnas_tabla = ["type", "quality", "alcohol", "pH", "residual sugar", "density", "volatile acidity", "citric acid", "chlorides", "sulphates", "free sulfur dioxide", "total sulfur dioxide"]
    st.dataframe(
        filtrado[columnas_tabla].head(10) if mostrar_todo else filtrado[columnas_tabla],
        width="stretch",
    )

    st.subheader("Estadisticas del conjunto filtrado")
    if filtrado.empty:
        st.warning("No hay registros con los filtros actuales.")
        return

    estadisticas = estadisticas_filtradas(filtrado)
    st.dataframe(estadisticas, width="stretch")

    st.subheader("Resumen de frecuencias de quality")
    freq_quality = filtrado["quality"].dropna()
    tabla_freq = tabla_frecuencias(freq_quality, bins=min(8, freq_quality.nunique())) if not freq_quality.empty else pd.DataFrame()
    st.dataframe(tabla_freq, width="stretch")

    st.subheader("Graficos dinamicos")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(grafico_tipo_vino(filtrado), width="stretch")
        _describir_grafico(
            "Grafico 1: tipos de vino",
            "Muestra cuantas muestras quedan de cada tipo despues de aplicar los filtros. Sirve para verificar si el recorte deja mas vino blanco o tinto dentro del analisis.",
        )
    with c2:
        st.plotly_chart(grafico_histograma_alcohol(filtrado), width="stretch")
        _describir_grafico(
            "Grafico 2: alcohol",
            "El histograma resume como se distribuye el alcohol en el conjunto filtrado. Cambia con todos los filtros porque solo usa los registros que cumplen las condiciones elegidas.",
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(grafico_boxplot_calidad_por_tipo(filtrado), width="stretch")
        _describir_grafico(
            "Grafico 3: quality por tipo",
            "El boxplot compara la calidad entre tipos de vino. Permite ver mediana, dispersion y valores atipicos bajo los filtros aplicados.",
        )
    with c4:
        st.plotly_chart(grafico_dispersion(filtrado), width="stretch")
        _describir_grafico(
            "Grafico 4: residual sugar vs alcohol",
            "La dispersion relaciona azucar residual con alcohol y colorea los puntos por quality. Ayuda a detectar si los vinos filtrados muestran una tendencia entre ambas variables.",
        )

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(grafico_correlation(filtrado), width="stretch")
        _describir_grafico(
            "Grafico 5: correlacion con quality",
            "Las barras muestran que variables numericas se mueven mas junto con quality en el subconjunto filtrado. Esto hace mas facil explicar que factores suben o bajan la calidad observada.",
        )
    with c6:
        st.plotly_chart(grafico_pH_por_tipo(filtrado), width="stretch")
        _describir_grafico(
            "Grafico 6: pH por tipo",
            "Este grafico compara el pH promedio entre tipos de vino. Es util porque el pH cambia con la acidez y ayuda a explicar diferencias de perfil entre blancos y tintos.",
        )

    st.subheader("Como leer el dashboard")
    st.write(
        "Los filtros de la barra lateral recortan el dataset original y todos los indicadores, tablas y graficos se recalculan automaticamente con los registros restantes. Primero se observa si el conjunto queda balanceado por tipo de vino, luego la calidad y el alcohol, y por ultimo se revisan las relaciones entre azucar, pH y correlaciones con quality."
    )


if __name__ == "__main__":
    main()
