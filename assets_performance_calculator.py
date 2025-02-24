# Paso 0: Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from google.colab import files

# ==========================
# Configuración del script
# ==========================
# Definir columnas de agrupación
GROUP_COLUMNS = ['CAMPAIGN', 'STAGE', 'PLATFORM', 'FORMAT']

# Definir columnas a normalizar
COLUMNS_TO_NORMALIZE = ['CAMPAIGN', 'STAGE', 'PLATFORM', 'FORMAT', 'BRAND', 'CATEGORY', 'PURCHASE_TYPE', 'CREATIVE_NAME', 'AUDIENCE']

# Lista de formatos de video
VIDEO_FORMATS = {
    'bumper', 'carousel_video', 'in_feed_video', 'instagram_feed_video',
    'instagram_reels', 'instagram_stories', 'online_video', 'page_post_video_ad',
    'placement_optimization_video', 'social_video', 'stories_video',
    'stories_video_carrousel', 'tiktok_video', 'trueview', 'topview',
    'youtube_for_reach', 'youtube_masthead', 'youtube_non_skippable',
    'youtube_shorts', 'youtube_skippable'
}

# Definir pesos específicos para cada combinación de tipo de compra y formato (video/estático)
WEIGHTS_BY_TYPE_AND_FORMAT = {
    ('cpa', 'video'): {'IMPRESSIONS': 0.15, 'QCPM': 0.2, 'VIEWABILITY': 0.25, 'CVTR': 0.15, 'CTR': 0.15, 'ER': 0.1},
    ('cpa', 'static'): {'IMPRESSIONS': 0.25, 'QCPM': 0.3, 'VIEWABILITY': 0.35, 'CVTR': 0.0, 'CTR': 0.1, 'ER': 0.0},
    ('cpc', 'video'): {'IMPRESSIONS': 0.1, 'QCPM': 0.1, 'VIEWABILITY': 0.2, 'CVTR': 0.4, 'CTR': 0.15, 'ER': 0.05},
    ('cpc', 'static'): {'IMPRESSIONS': 0.2, 'QCPM': 0.15, 'VIEWABILITY': 0.3, 'CVTR': 0.0, 'CTR': 0.3, 'ER': 0.05},
    ('cpcv', 'video'): {'IMPRESSIONS': 0.1, 'QCPM': 0.2, 'VIEWABILITY': 0.3, 'CVTR': 0.3, 'CTR': 0.05, 'ER': 0.05},
    ('cpcv', 'static'): {'IMPRESSIONS': 0.2, 'QCPM': 0.25, 'VIEWABILITY': 0.35, 'CVTR': 0.0, 'CTR': 0.15, 'ER': 0.05},
    ('cpl', 'video'): {'IMPRESSIONS': 0.1, 'QCPM': 0.15, 'VIEWABILITY': 0.25, 'CVTR': 0.35, 'CTR': 0.1, 'ER': 0.05},
    ('cpl', 'static'): {'IMPRESSIONS': 0.3, 'QCPM': 0.3, 'VIEWABILITY': 0.25, 'CVTR': 0.0, 'CTR': 0.1, 'ER': 0.05},
    ('cpm', 'video'): {'IMPRESSIONS': 0.2, 'QCPM': 0.25, 'VIEWABILITY': 0.25, 'CVTR': 0.15, 'CTR': 0.1, 'ER': 0.05},
    ('cpm', 'static'): {'IMPRESSIONS': 0.35, 'QCPM': 0.3, 'VIEWABILITY': 0.2, 'CVTR': 0.0, 'CTR': 0.1, 'ER': 0.05},
    ('cpv', 'video'): {'IMPRESSIONS': 0.15, 'QCPM': 0.2, 'VIEWABILITY': 0.3, 'CVTR': 0.25, 'CTR': 0.05, 'ER': 0.05},
    ('cpv', 'static'): {'IMPRESSIONS': 0.25, 'QCPM': 0.25, 'VIEWABILITY': 0.3, 'CVTR': 0.0, 'CTR': 0.15, 'ER': 0.05},
    ('top_view', 'video'): {'IMPRESSIONS': 0.15, 'QCPM': 0.15, 'VIEWABILITY': 0.35, 'CVTR': 0.25, 'CTR': 0.05, 'ER': 0.05},
    ('top_view', 'static'): {'IMPRESSIONS': 0.2, 'QCPM': 0.2, 'VIEWABILITY': 0.4, 'CVTR': 0.0, 'CTR': 0.15, 'ER': 0.05},
    'default': {'IMPRESSIONS': 0.2, 'QCPM': 0.2, 'VIEWABILITY': 0.2, 'CVTR': 0.2, 'CTR': 0.1, 'ER': 0.1}
}

# Factor de penalización para Quality_Impressions menores a 1000
PENALTY_FACTOR = 1.5

# ==========================
# Funciones de utilidad
# ==========================
def normalize_text(series):
    """Normaliza el texto eliminando espacios, pasando a minúsculas y reemplazando espacios con guiones bajos."""
    return series.str.strip().str.lower().str.replace(' ', '_')

def convert_series(series):
    """Convierte valores de series a numéricos, eliminando separadores de miles y convirtiendo porcentajes."""
    if series.dtype == 'object':
        series = series.fillna('0')  # Reemplazar celdas vacías por '0'
        series = series.str.replace(',', '')  # Eliminar separadores de miles
        series = series.apply(lambda x: float(x.replace('%', '')) / 100 if isinstance(x, str) and '%' in x else x)
    return pd.to_numeric(series, errors='coerce').fillna(0)  # Convertir a numérico y manejar NaN

def compute_index(df, score_column):
    """Calcula el performance index usando fórmula min-max dentro de grupos definidos."""
    epsilon = 1e-7
    df['performance_index'] = df.groupby(GROUP_COLUMNS)[score_column].transform(
        lambda x: 1.0 if (x.max() == x.min()) else (1 - (x - x.min()) / ((x.max() - x.min()) + epsilon))
    )
    return df

def print_nan_info_before_after(df, file_name, numeric_columns):
    """Evaluar NaN antes y después de la transformación y descarga opcional de archivo."""
    print(f"\nEvaluación de NaN en {file_name}:")
    nan_info_before = df[numeric_columns].isna().sum()
    print("Valores NaN antes de la conversión:")
    for column, num_nan in nan_info_before.items():
        print(f"- Columna {column}: {num_nan} valores NaN")

    df[numeric_columns] = df[numeric_columns].apply(convert_series)

    nan_info_after = df[numeric_columns].isna().sum()
    print("Valores NaN después de la conversión:")
    for column, num_nan in nan_info_after.items():
        print(f"- Columna {column}: {num_nan} valores NaN")

def calculate_final_score(row):
    """Calcula el score final usando los pesos específicos según el tipo de compra y el formato."""
    # Determinar el formato del contenido: "video" o "static"
    format_type = 'video' if row['FORMAT'] in VIDEO_FORMATS else 'static'

    # Obtener pesos según el tipo de compra y formato
    weights = WEIGHTS_BY_TYPE_AND_FORMAT.get((row['PURCHASE_TYPE'], format_type), WEIGHTS_BY_TYPE_AND_FORMAT['default'])

    return (
        row['Quality_Impressions_rank'] * weights['IMPRESSIONS'] +
        row['QCPM_combined'] * weights['QCPM'] +
        row['VIEWABILITY_combined'] * weights['VIEWABILITY'] +
        row['CVTR_combined'] * weights['CVTR'] +
        row['CTR_combined'] * weights['CTR'] +
        row['ER_combined'] * weights['ER']
    )

# ==========================
# Procesamiento de datos
# ==========================
# Paso 1: Subir los archivos CSV
print("Sube el archivo CSV con los datos de campaña:")
uploaded_campaign = files.upload()
campaign_file = list(uploaded_campaign.keys())[0]
campaign_df = pd.read_csv(campaign_file)

print("Sube el archivo CSV con los benchmarks:")
uploaded_bench = files.upload()
bench_file = list(uploaded_bench.keys())[0]
benchmark_df = pd.read_csv(bench_file)

# Normalizar los valores de las columnas relevantes en el DataFrame de campaña
campaign_df[COLUMNS_TO_NORMALIZE] = campaign_df[COLUMNS_TO_NORMALIZE].apply(normalize_text)

# Normalizar solo las columnas presentes en ambos DataFrames
common_columns = benchmark_df.columns.intersection(COLUMNS_TO_NORMALIZE)
benchmark_df[common_columns] = benchmark_df[common_columns].apply(normalize_text)

# Definición de columnas numéricas
campaign_numeric_cols = [
    'IMPRESSIONS', 'VIDEO_VIEWS', 'COMPLETE_VIEWS', 'CLICS', 'COMMENTS',
    'INTERACTIONS', 'SHARES', 'REACH', 'MEDIA_SPEND', 'CPM', 'VTR', 'CVTR', 'CTR', 'ER', 'VIEWABILITY'
]
bench_numeric_cols = ['QCPM', 'VIEWABILITY', 'CVTR', 'CTR', 'ER']

# Conversión de columnas y evaluación de NaN
print_nan_info_before_after(campaign_df, "de campaña", campaign_numeric_cols)
print_nan_info_before_after(benchmark_df, "de benchmarks", bench_numeric_cols)

# Cálculos adicionales y combinaciones
campaign_df['Quality_Impressions'] = campaign_df['IMPRESSIONS'] * campaign_df['VIEWABILITY']
campaign_df['QCPM_calculated'] = np.where(
    campaign_df['Quality_Impressions'] != 0,
    (campaign_df['MEDIA_SPEND'] / campaign_df['Quality_Impressions']) * 1000,
    0
)

df = pd.merge(campaign_df, benchmark_df, on=['PLATFORM', 'STAGE'], how='left', suffixes=("", "_bench"))

# Cálculo de ratios y rankings
df['QCPM_ratio'] = np.where(df['QCPM_calculated'] != 0, df['QCPM'] / df['QCPM_calculated'], 0)
df['VIEWABILITY_ratio'] = np.where(df['VIEWABILITY_bench'] != 0, df['VIEWABILITY'] / df['VIEWABILITY_bench'], 0)
df['CVTR_ratio'] = np.where(df['CVTR_bench'] != 0, df['CVTR'] / df['CVTR_bench'], 0)
df['CTR_ratio'] = np.where(df['CTR_bench'] != 0, df['CTR'] / df['CTR_bench'], 0)
df['ER_ratio'] = np.where(df['ER_bench'] != 0, df['ER'] / df['ER_bench'], 0)

df['QCPM_rank'] = df.groupby(GROUP_COLUMNS)['QCPM_calculated'].rank(method='min', ascending=True)
df['Quality_Impressions_rank'] = df.groupby(GROUP_COLUMNS)['Quality_Impressions'].rank(method='min', ascending=False)
df['VIEWABILITY_rank'] = df.groupby(GROUP_COLUMNS)['VIEWABILITY'].rank(method='min', ascending=False)
df['CVTR_rank'] = df.groupby(GROUP_COLUMNS)['CVTR'].rank(method='min', ascending=False)
df['CTR_rank'] = df.groupby(GROUP_COLUMNS)['CTR'].rank(method='min', ascending=False)
df['ER_rank'] = df.groupby(GROUP_COLUMNS)['ER'].rank(method='min', ascending=False)

# Combinación de rankings y ratios
df['QCPM_combined'] = 0.5 * df['QCPM_rank'] + 0.5 * df['QCPM_ratio']
df['VIEWABILITY_combined'] = 0.5 * df['VIEWABILITY_rank'] + 0.5 * df['VIEWABILITY_ratio']
df['CVTR_combined'] = 0.5 * df['CVTR_rank'] + 0.5 * df['CVTR_ratio']
df['CTR_combined'] = 0.5 * df['CTR_rank'] + 0.5 * df['CTR_ratio']
df['ER_combined'] = 0.5 * df['ER_rank'] + 0.5 * df['ER_ratio']

# Cálculo del score final
df['final_score'] = df.apply(calculate_final_score, axis=1)
df.loc[df['Quality_Impressions'] < 1000, 'final_score'] *= PENALTY_FACTOR  # Penalización

# Cálculo del índice de performance_index
df = compute_index(df, 'final_score')

# Redondear todas las columnas numéricas a 2 decimales
numeric_cols_to_round = [
    'IMPRESSIONS', 'VIDEO_VIEWS', 'COMPLETE_VIEWS', 'CLICS', 'COMMENTS',
    'INTERACTIONS', 'SHARES', 'REACH', 'MEDIA_SPEND', 'CPM', 'VTR',
    'CVTR', 'CTR', 'ER', 'VIEWABILITY', 'Quality_Impressions',
    'QCPM_calculated', 'QCPM_ratio', 'VIEWABILITY_ratio', 'CVTR_ratio',
    'CTR_ratio', 'ER_ratio', 'final_score', 'performance_index'
]
df[numeric_cols_to_round] = df[numeric_cols_to_round].round(2)

# ==========================
# Exportación de resultados - Hoja 1
# ==========================
base_columns = [
    'MONTH', 'PLATFORM', 'CREATIVE_ID', 'CREATIVE_NAME', 'CAMPAIGN', 'BRAND',
    'STAGE', 'AUDIENCE', 'FORMAT', 'CATEGORY', 'PURCHASE_TYPE',
    'IMPRESSIONS', 'VIDEO_VIEWS', 'COMPLETE_VIEWS', 'CLICS', 'COMMENTS',
    'INTERACTIONS', 'SHARES', 'REACH', 'MEDIA_SPEND', 'CPM', 'VTR',
    'CVTR', 'CTR', 'ER', 'VIEWABILITY'
]
result_columns = base_columns + [
    'Quality_Impressions', 'QCPM_calculated', 'QCPM_ratio', 'VIEWABILITY_ratio',
    'CVTR_ratio', 'CTR_ratio', 'ER_ratio', 'Quality_Impressions_rank',
    'QCPM_rank', 'VIEWABILITY_rank', 'CVTR_rank', 'CTR_rank', 'ER_rank',
    'final_score', 'performance_index'
]

result_df = df[result_columns]

# ==========================
# Calcular Z-scores usando Median y MAD - Hoja 2
# ==========================
# Calcular la Mediana de `final_scores` para cada campaña
campaign_median_summary = df.groupby('CAMPAIGN')['final_score'].median().reset_index()
campaign_median_summary = campaign_median_summary.rename(columns={'final_score': 'median_score'})

# Cálculo del MAD (Desviación Absoluta Mediana)
mad = np.median(np.abs(campaign_median_summary['median_score'] - campaign_median_summary['median_score'].median()))

# Calcular z-score basado en Mediana y MAD
campaign_median_summary['z_score'] = (campaign_median_summary['median_score'] - campaign_median_summary['median_score'].median()) / mad

# Exportación de resultados a Excel con dos hojas
with pd.ExcelWriter("final_results.xlsx", engine='openpyxl') as writer:
    # Hoja 1: Resultados por Asset
    result_df.to_excel(writer, sheet_name='Resultados por Asset', index=False)

    # Hoja 2: Resumen de Campañas basado en Mediana
    campaign_median_summary.to_excel(writer, sheet_name='Resumen de Campañas', index=False)

print("El archivo final ha sido guardado como 'final_results.xlsx'")
files.download("final_results.xlsx")
