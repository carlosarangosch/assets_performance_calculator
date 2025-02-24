# ==========================
# Importación de bibliotecas
# ==========================
import pandas as pd
import numpy as np
from google.colab import files

# ==========================
# Funciones de Utilidad
# ==========================

def normalize_text(series):
    """
    Normaliza el texto eliminando espacios al inicio y final, convirtiendo a minúsculas 
    y reemplazando espacios intermedios con guiones bajos.
    """
    return series.str.strip().str.lower().str.replace(' ', '_')

def convert_series(series):
    """
    Convierte los valores de una serie a numéricos.
    - Rellena los valores nulos con '0'
    - Elimina separadores de miles
    - Convierte cadenas con porcentajes dividiéndolos por 100
    - Finalmente, convierte la serie a tipo numérico y reemplaza NaN por 0.
    """
    if series.dtype == 'object':
        series = series.fillna('0')
        series = series.str.replace(',', '')
        series = series.apply(lambda x: float(x.replace('%', '')) / 100 if isinstance(x, str) and '%' in x else x)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def print_nan_info_before_after(df, file_name, numeric_columns):
    """
    Muestra la cantidad de valores NaN en las columnas numéricas antes y después de la conversión.
    """
    print(f"\nEvaluación de NaN en {file_name}:")
    nan_info_before = df[numeric_columns].isna().sum()
    print("Valores NaN antes de la conversión:")
    for column, num_nan in nan_info_before.items():
        print(f"- Columna {column}: {num_nan} valores NaN")

    # Aplicar conversión a numérico
    df[numeric_columns] = df[numeric_columns].apply(convert_series)

    nan_info_after = df[numeric_columns].isna().sum()
    print("Valores NaN después de la conversión:")
    for column, num_nan in nan_info_after.items():
        print(f"- Columna {column}: {num_nan} valores NaN")

def mad(series):
    """
    Calcula la Desviación Absoluta Mediana (MAD) de una serie.
    MAD se define como la mediana de las diferencias absolutas entre cada valor y la mediana.
    """
    med = np.median(series)
    return np.median(np.abs(series - med))

# ==========================
# Procesamiento de Datos
# ==========================

# Paso 1: Cargar el archivo CSV con los datos de campaña
print("Sube el archivo CSV con los datos de campaña:")
uploaded_campaign = files.upload()
campaign_file = list(uploaded_campaign.keys())[0]
campaign_df = pd.read_csv(campaign_file)

# Paso 2: Normalizar las columnas de texto relevantes
columns_to_normalize = ['PLATFORM', 'STAGE', 'FORMAT', 'BRAND', 'CATEGORY', 'PURCHASE_TYPE', 'CREATIVE_NAME', 'AUDIENCE']
campaign_df[columns_to_normalize] = campaign_df[columns_to_normalize].apply(normalize_text)

# Paso 3: Convertir las columnas numéricas a formato numérico
numeric_columns = [
    'IMPRESSIONS', 'VIDEO_VIEWS', 'COMPLETE_VIEWS', 'CLICS', 'COMMENTS',
    'INTERACTIONS', 'SHARES', 'REACH', 'MEDIA_SPEND', 'CPM', 'VTR', 'CVTR', 'CTR', 'ER', 'VIEWABILITY'
]
print_nan_info_before_after(campaign_df, "de campaña", numeric_columns)

# Paso 4: Calcular métricas adicionales
# Quality Impressions pondera las impresiones por la viewability para obtener una medida de calidad.
campaign_df['Quality_Impressions'] = campaign_df['IMPRESSIONS'] * campaign_df['VIEWABILITY']

# QCPM_calculated: Costo por mil impresiones de calidad. Se evita la división por cero.
campaign_df['QCPM_calculated'] = np.where(
    campaign_df['Quality_Impressions'] != 0,
    (campaign_df['MEDIA_SPEND'] / campaign_df['Quality_Impressions']) * 1000,
    0
)

# ==========================
# Agrupación y Cálculo de Benchmarks
# ==========================

# Definir las columnas de agrupación
group_cols = ['PLATFORM', 'STAGE', 'FORMAT']

# Lista de métricas para las cuales se calcularán la mediana y el MAD
# Incluye QCPM_calculated para que se trate de la misma forma que las demás métricas.
metrics = ['CPM', 'VIEWABILITY', 'CVTR', 'CTR', 'ER', 'QCPM_calculated']

# Crear un diccionario para la agregación:
# Para cada métrica se calculará la mediana y el MAD.
agg_dict = {}
for metric in metrics:
    agg_dict[metric] = ['median', mad]

# Agrupar el DataFrame original y aplicar las funciones de agregación definidas
benchmarks = campaign_df.groupby(group_cols).agg(agg_dict)

# Aplanar las columnas del DataFrame resultante (MultiIndex) para obtener nombres simples
benchmarks.columns = ['_'.join(col).strip() for col in benchmarks.columns.values]
benchmarks = benchmarks.reset_index()

# Renombrar la columna del MAD para que tenga el formato 'MAD_<Métrica>'
for metric in metrics:
    benchmarks.rename(columns={
        f"{metric}_mad": f"MAD_{metric}"
    }, inplace=True)

# Calcular las métricas ajustadas sumando la mediana y el MAD para cada métrica
for metric in metrics:
    benchmarks[f"adjusted_{metric}"] = benchmarks[f"{metric}_median"] + benchmarks[f"MAD_{metric}"]

# ==========================
# Preparación de Datos para Exportación
# ==========================

# Hoja 1: Benchmark Ajustado (solo columnas de valores ajustados)
# Se incluye también las columnas de agrupación.
adjusted_cols = [f"adjusted_{metric}" for metric in metrics]
benchmark_adjusted = benchmarks[group_cols + adjusted_cols].copy()
# Renombrar las columnas ajustadas para quitar el prefijo 'adjusted_'
rename_dict = {f"adjusted_{metric}": metric for metric in metrics}
benchmark_adjusted.rename(columns=rename_dict, inplace=True)

# Hoja 2: Benchmark Completo (con columnas organizadas: mediana, MAD y ajustado para cada métrica)
# Se crea un listado ordenado de columnas: primero las de agrupación y luego para cada métrica sus tres columnas.
ordered_cols = group_cols.copy()
for metric in metrics:
    ordered_cols.extend([f"{metric}_median", f"MAD_{metric}", f"adjusted_{metric}"])
benchmark_complete = benchmarks[ordered_cols].copy()

# ==========================
# Exportación a Excel
# ==========================
output_file = "benchmarks_results.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Exportar hoja 1: Benchmark Ajustado y limpio
    benchmark_adjusted.to_excel(writer, sheet_name='Benchmark Ajustado', index=False)
    # Exportar hoja 2: Benchmark Completo y organizado
    benchmark_complete.to_excel(writer, sheet_name='Benchmark Completo', index=False)

print(f"\nEl archivo de benchmarks ha sido guardado como '{output_file}'")
files.download(output_file)
