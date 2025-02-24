# =============================================================================
# METODOLOGÍA INTEGRAL: Evaluación de Assets y Campañas
#
# Esta metodología evalúa cada asset considerando:
# 1. Desempeño interno (ranking dentro de la campaña, stage, plataforma y formato).
# 2. Comparación frente a benchmarks externos.
# 3. Pesos específicos según el PURCHASE_TYPE y formato (video o estático).
# 4. Ajuste que prioriza assets con mayor volumen de impresiones.
#
# Además, se agruparán las campañas en función de:
#   - Inversión Total: Baja (20–100M), Media (101–500M), Alta (501M en adelante).
#   - Cantidad de Formatos: Poca Variedad (1–3), Variedad Moderada (4–10), Gran Variedad (11 en adelante).
#   - Cantidad de Plataformas: Cobertura Limitada (1–2), Cobertura Moderada (3–4), Cobertura Amplia (5 en adelante).
#
# Se exportarán dos hojas en Excel:
#   - Hoja 1: Resultados por Asset (incluye final_score y performance_index, además de los rankings raw).
#   - Hoja 2: Resumen de Campañas (estadísticos robustos segmentados por grupo y la etiqueta de grupo).
# =============================================================================

# =============================================================================
# Paso 0: Importar Bibliotecas y Configuración Inicial
# =============================================================================
import pandas as pd
import numpy as np
from google.colab import files

# Columnas para agrupar assets
GROUP_COLUMNS = ['CAMPAIGN', 'STAGE', 'PLATFORM', 'FORMAT']

# Columnas de texto a normalizar
COLUMNS_TO_NORMALIZE = [
    'CAMPAIGN', 'STAGE', 'PLATFORM', 'FORMAT',
    'BRAND', 'CATEGORY', 'PURCHASE_TYPE', 'CREATIVE_NAME', 'AUDIENCE'
]

# Lista de formatos de video
VIDEO_FORMATS = {
    'bumper', 'carousel_video', 'in_feed_video', 'instagram_feed_video',
    'instagram_reels', 'instagram_stories', 'online_video', 'page_post_video_ad',
    'placement_optimization_video', 'social_video', 'stories_video',
    'stories_video_carrousel', 'tiktok_video', 'trueview', 'topview',
    'youtube_for_reach', 'youtube_masthead', 'youtube_non_skippable',
    'youtube_shorts', 'youtube_skippable'
}

# =============================================================================
# Nuevo Diccionario de Pesos (sin IMPRESSIONS; suma de pesos = 1, escala de 5 en múltiplos de 0.125)
# =============================================================================
WEIGHTS_BY_TYPE_AND_FORMAT = {
    ('cpa', 'video'): {'QCPM': 0.25, 'VIEWABILITY': 0.25, 'CVTR': 0.25, 'CTR': 0.125, 'ER': 0.125},
    ('cpa', 'static'): {'QCPM': 0.375, 'VIEWABILITY': 0.5, 'CVTR': 0.0, 'CTR': 0.125, 'ER': 0.0},
    ('cpc', 'video'): {'QCPM': 0.125, 'VIEWABILITY': 0.25, 'CVTR': 0.5, 'CTR': 0.125, 'ER': 0.0},
    ('cpc', 'static'): {'QCPM': 0.25, 'VIEWABILITY': 0.375, 'CVTR': 0.0, 'CTR': 0.25, 'ER': 0.125},
    ('cpcv', 'video'): {'QCPM': 0.25, 'VIEWABILITY': 0.375, 'CVTR': 0.375, 'CTR': 0.0, 'ER': 0.0},
    ('cpcv', 'static'): {'QCPM': 0.375, 'VIEWABILITY': 0.375, 'CVTR': 0.0, 'CTR': 0.125, 'ER': 0.125},
    ('cpl', 'video'): {'QCPM': 0.125, 'VIEWABILITY': 0.25, 'CVTR': 0.375, 'CTR': 0.125, 'ER': 0.125},
    ('cpl', 'static'): {'QCPM': 0.5, 'VIEWABILITY': 0.375, 'CVTR': 0.0, 'CTR': 0.125, 'ER': 0.0},
    ('cpm', 'video'): {'QCPM': 0.375, 'VIEWABILITY': 0.375, 'CVTR': 0.125, 'CTR': 0.125, 'ER': 0.0},
    ('cpm', 'static'): {'QCPM': 0.5, 'VIEWABILITY': 0.375, 'CVTR': 0.0, 'CTR': 0.125, 'ER': 0.0},
    ('cpv', 'video'): {'QCPM': 0.25, 'VIEWABILITY': 0.375, 'CVTR': 0.25, 'CTR': 0.125, 'ER': 0.0},
    ('cpv', 'static'): {'QCPM': 0.375, 'VIEWABILITY': 0.375, 'CVTR': 0.0, 'CTR': 0.25, 'ER': 0.0},
    ('top_view', 'video'): {'QCPM': 0.25, 'VIEWABILITY': 0.375, 'CVTR': 0.25, 'CTR': 0.125, 'ER': 0.0},
    ('top_view', 'static'): {'QCPM': 0.25, 'VIEWABILITY': 0.5, 'CVTR': 0.0, 'CTR': 0.125, 'ER': 0.125},
    'default': {'QCPM': 0.25, 'VIEWABILITY': 0.25, 'CVTR': 0.25, 'CTR': 0.125, 'ER': 0.125}
}

# =============================================================================
# Funciones Globales de Normalización
# =============================================================================
def min_max_normalize(series):
    """
    Normaliza una serie numérica usando Min-Max:
    (x - min) / (max - min). Si max == min, retorna 0.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series.apply(lambda x: 0)
    return (series - min_val) / (max_val - min_val)

def normalize_ranking(series):
    """
    Normaliza una serie de rankings usando:
      ranking_norm = 1 - ((rank - 1) / (N - 1))
    donde N es el número de elementos. Si N <= 1, retorna 1.
    """
    N = series.count()
    if N <= 1:
        return series.apply(lambda x: 1)
    return 1 - ((series - 1) / (N - 1))

# =============================================================================
# Otras Funciones de Utilidad
# =============================================================================
def normalize_text(series):
    """Normaliza el texto: elimina espacios, convierte a minúsculas y reemplaza espacios por guiones bajos."""
    return series.str.strip().str.lower().str.replace(' ', '_')

def convert_series(series):
    """Convierte una serie a valores numéricos: elimina separadores de miles y convierte porcentajes."""
    if series.dtype == 'object':
        series = series.fillna('0')
        series = series.str.replace(',', '')
        series = series.apply(lambda x: float(x.replace('%', '')) / 100 if isinstance(x, str) and '%' in x else x)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def compute_index(df, score_column):
    """
    Calcula el Performance Index normalizando el final_score (que ya incluye
    el factor de impresiones) dentro de cada grupo definido por GROUP_COLUMNS.
    Se utiliza una normalización directa, de modo que el asset con el mayor
    final_score obtenga un performance_index de 1.
    """
    epsilon = 1e-7
    df['performance_index'] = df.groupby(GROUP_COLUMNS)[score_column].transform(
        lambda x: (x - x.min()) / ((x.max() - x.min()) + epsilon)
    )
    return df

def print_nan_info_before_after(df, file_name, numeric_columns):
    """Imprime información de NaN antes y después de la conversión numérica."""
    print(f"\nEvaluación de NaN en {file_name}:")
    nan_info_before = df[numeric_columns].isna().sum()
    print("Valores NaN antes de la conversión:")
    for col, num in nan_info_before.items():
        print(f"- {col}: {num}")
    df[numeric_columns] = df[numeric_columns].apply(convert_series)
    nan_info_after = df[numeric_columns].isna().sum()
    print("Valores NaN después de la conversión:")
    for col, num in nan_info_after.items():
        print(f"- {col}: {num}")

def calculate_final_score(row):
    """
    Calcula el Puntaje Final de un asset aplicando los pesos específicos según
    el PURCHASE_TYPE y el formato (video o static). Se omite Quality_Impressions
    en la suma, pues su impacto se refleja mediante el factor de impresiones.
    """
    format_type = 'video' if row['FORMAT'] in VIDEO_FORMATS else 'static'
    weights = WEIGHTS_BY_TYPE_AND_FORMAT.get((row['PURCHASE_TYPE'], format_type),
                                               WEIGHTS_BY_TYPE_AND_FORMAT['default'])
    return (
        row['QCPM_combined'] * weights['QCPM'] +
        row['VIEWABILITY_combined'] * weights['VIEWABILITY'] +
        row['CVTR_combined'] * weights['CVTR'] +
        row['CTR_combined'] * weights['CTR'] +
        row['ER_combined'] * weights['ER']
    )

# =============================================================================
# 1. Preprocesamiento y Normalización de Datos
# =============================================================================
print("Sube el archivo CSV con los datos de campaña:")
uploaded_campaign = files.upload()
campaign_file = list(uploaded_campaign.keys())[0]
campaign_df = pd.read_csv(campaign_file)

print("Sube el archivo CSV con los benchmarks:")
uploaded_bench = files.upload()
bench_file = list(uploaded_bench.keys())[0]
benchmark_df = pd.read_csv(bench_file)

# Normalizar columnas de texto
campaign_df[COLUMNS_TO_NORMALIZE] = campaign_df[COLUMNS_TO_NORMALIZE].apply(normalize_text)
common_columns = benchmark_df.columns.intersection(COLUMNS_TO_NORMALIZE)
benchmark_df[common_columns] = benchmark_df[common_columns].apply(normalize_text)

# Definir columnas numéricas
campaign_numeric_cols = [
    'IMPRESSIONS', 'VIDEO_VIEWS', 'COMPLETE_VIEWS', 'CLICS', 'COMMENTS',
    'INTERACTIONS', 'SHARES', 'REACH', 'MEDIA_SPEND', 'CPM', 'VTR',
    'CVTR', 'CTR', 'ER', 'VIEWABILITY'
]
bench_numeric_cols = ['QCPM', 'VIEWABILITY', 'CVTR', 'CTR', 'ER']

print_nan_info_before_after(campaign_df, "de campaña", campaign_numeric_cols)
print_nan_info_before_after(benchmark_df, "de benchmarks", bench_numeric_cols)

# Calcular métricas derivadas
campaign_df['Quality_Impressions'] = campaign_df['IMPRESSIONS'] * campaign_df['VIEWABILITY']
campaign_df['QCPM_calculated'] = np.where(
    campaign_df['Quality_Impressions'] != 0,
    (campaign_df['MEDIA_SPEND'] / campaign_df['Quality_Impressions']) * 1000,
    0
)

# =============================================================================
# 2. Fusión de Datos y Cálculo de Ratios con Benchmarks
# =============================================================================
# CAMBIO: Se añade 'FORMAT' en la fusión para alinear con el nuevo benchmark segmentado por PLATFORM, STAGE y FORMAT
df = pd.merge(campaign_df, benchmark_df, on=['PLATFORM', 'STAGE', 'FORMAT'], how='left', suffixes=("", "_bench"))

# Calcular ratios:
# Para métricas donde "menor es mejor" (ej. QCPM): Ratio = Benchmark / Valor asset
df['QCPM_ratio'] = np.where(df['QCPM_calculated'] != 0, df['QCPM'] / df['QCPM_calculated'], 0)
# Para métricas donde "mayor es mejor" (ej. VIEWABILITY, CVTR, CTR, ER): Ratio = Valor asset / Benchmark
df['VIEWABILITY_ratio'] = np.where(df['VIEWABILITY_bench'] != 0, df['VIEWABILITY'] / df['VIEWABILITY_bench'], 0)
df['CVTR_ratio'] = np.where(df['CVTR_bench'] != 0, df['CVTR'] / df['CVTR_bench'], 0)
df['CTR_ratio'] = np.where(df['CTR_bench'] != 0, df['CTR'] / df['CTR_bench'], 0)
df['ER_ratio'] = np.where(df['ER_bench'] != 0, df['ER'] / df['ER_bench'], 0)

# Normalizar los ratios usando Min-Max por grupo
df['QCPM_ratio_norm'] = df.groupby(GROUP_COLUMNS)['QCPM_ratio'].transform(min_max_normalize)
df['VIEWABILITY_ratio_norm'] = df.groupby(GROUP_COLUMNS)['VIEWABILITY_ratio'].transform(min_max_normalize)
df['CVTR_ratio_norm'] = df.groupby(GROUP_COLUMNS)['CVTR_ratio'].transform(min_max_normalize)
df['CTR_ratio_norm'] = df.groupby(GROUP_COLUMNS)['CTR_ratio'].transform(min_max_normalize)
df['ER_ratio_norm'] = df.groupby(GROUP_COLUMNS)['ER_ratio'].transform(min_max_normalize)

# =============================================================================
# 3. Cálculo del Ranking Interno (Contexto de Campaña)
# =============================================================================
# Asignar ranking raw por cada métrica dentro de GROUP_COLUMNS
df['QCPM_rank_raw'] = df.groupby(GROUP_COLUMNS)['QCPM_calculated'].rank(method='min', ascending=True)
df['Quality_Impressions_rank_raw'] = df.groupby(GROUP_COLUMNS)['Quality_Impressions'].rank(method='min', ascending=False)
df['VIEWABILITY_rank_raw'] = df.groupby(GROUP_COLUMNS)['VIEWABILITY'].rank(method='min', ascending=False)
df['CVTR_rank_raw'] = df.groupby(GROUP_COLUMNS)['CVTR'].rank(method='min', ascending=False)
df['CTR_rank_raw'] = df.groupby(GROUP_COLUMNS)['CTR'].rank(method='min', ascending=False)
df['ER_rank_raw'] = df.groupby(GROUP_COLUMNS)['ER'].rank(method='min', ascending=False)

# Normalizar los rankings usando la función global (por grupo)
df['QCPM_rank_norm'] = df.groupby(GROUP_COLUMNS)['QCPM_rank_raw'].transform(normalize_ranking)
df['Quality_Impressions_rank_norm'] = df.groupby(GROUP_COLUMNS)['Quality_Impressions_rank_raw'].transform(normalize_ranking)
df['VIEWABILITY_rank_norm'] = df.groupby(GROUP_COLUMNS)['VIEWABILITY_rank_raw'].transform(normalize_ranking)
df['CVTR_rank_norm'] = df.groupby(GROUP_COLUMNS)['CVTR_rank_raw'].transform(normalize_ranking)
df['CTR_rank_norm'] = df.groupby(GROUP_COLUMNS)['CTR_rank_raw'].transform(normalize_ranking)
df['ER_rank_norm'] = df.groupby(GROUP_COLUMNS)['ER_rank_raw'].transform(normalize_ranking)

# =============================================================================
# 4. Combinación de Ranking Interno y Ratios Benchmark a Nivel de Métrica
# =============================================================================
# Combinar cada métrica: 50% ranking normalizado + 50% ratio normalizado
df['QCPM_combined'] = 0.5 * df['QCPM_rank_norm'] + 0.5 * df['QCPM_ratio_norm']
df['VIEWABILITY_combined'] = 0.5 * df['VIEWABILITY_rank_norm'] + 0.5 * df['VIEWABILITY_ratio_norm']
df['CVTR_combined'] = 0.5 * df['CVTR_rank_norm'] + 0.5 * df['CVTR_ratio_norm']
df['CTR_combined'] = 0.5 * df['CTR_rank_norm'] + 0.5 * df['CTR_ratio_norm']
df['ER_combined'] = 0.5 * df['ER_rank_norm'] + 0.5 * df['ER_ratio_norm']

# =============================================================================
# 5. Aplicación de Pesos Específicos según Purchase Type y Formato
# =============================================================================
df['final_score'] = df.apply(calculate_final_score, axis=1)

# =============================================================================
# 6. Incorporación de un Factor de Impresiones
# =============================================================================
# Calcular el factor de impresiones: (IMPRESSIONS del asset) / (máximo de IMPRESSIONS en el grupo)
df['impressions_factor'] = df.groupby(GROUP_COLUMNS)['IMPRESSIONS'].transform(lambda x: x / x.max())
# Incorporar el factor de impresiones al final_score
df['final_score'] = df['final_score'] * df['impressions_factor']

# =============================================================================
# 7. Cálculo del Performance Index (Hoja 1)
# =============================================================================
df = compute_index(df, 'final_score')

# Redondear columnas numéricas para mayor claridad
numeric_cols_to_round = [
    'IMPRESSIONS', 'VIDEO_VIEWS', 'COMPLETE_VIEWS', 'CLICS', 'COMMENTS',
    'INTERACTIONS', 'SHARES', 'REACH', 'MEDIA_SPEND', 'CPM', 'VTR',
    'CVTR', 'CTR', 'ER', 'VIEWABILITY', 'Quality_Impressions',
    'QCPM_calculated', 'QCPM_ratio', 'VIEWABILITY_ratio', 'CVTR_ratio',
    'CTR_ratio', 'ER_ratio', 'final_score', 'performance_index'
]
df[numeric_cols_to_round] = df[numeric_cols_to_round].round(2)

# =============================================================================
# 8. Agrupación de Campañas para Hoja 2 (Segmentada por Grupo)
# =============================================================================
# Agrupar campañas para obtener:
# - Total_MEDIA_SPEND: suma de MEDIA_SPEND de todos los assets de la campaña.
# - Num_Formatos: número de formatos únicos usados en la campaña.
# - Num_Plataformas: número de plataformas únicos usados en la campaña.
campaign_groups = df.groupby('CAMPAIGN').agg({
    'MEDIA_SPEND': 'sum',
    'FORMAT': lambda x: x.nunique(),
    'PLATFORM': lambda x: x.nunique()
}).reset_index()

# Renombrar columnas para mayor claridad
campaign_groups.rename(columns={
    'MEDIA_SPEND': 'Total_MEDIA_SPEND',
    'FORMAT': 'Num_Formatos',
    'PLATFORM': 'Num_Plataformas'
}, inplace=True)

# Función para asignar grupo a cada campaña
def asignar_grupo(row):
    """
    Asigna un grupo a la campaña basado en:
      - Inversión Total (Total_MEDIA_SPEND) en pesos colombianos:
            Baja Inversión: 20,000,000 - 100,000,000
            Media Inversión: 101,000,000 - 500,000,000
            Alta Inversión: 501,000,000 en adelante
      - Cantidad de Formatos (Num_Formatos):
            Poca Variedad: 1–3
            Variedad Moderada: 4–10
            Gran Variedad: 11 en adelante
      - Cantidad de Plataformas (Num_Plataformas):
            Cobertura Limitada: 1–2
            Cobertura Moderada: 3–4
            Cobertura Amplia: 5 en adelante
    """
    inv = row['Total_MEDIA_SPEND']
    if 20000000 <= inv <= 100000000:
        grupo_inv = "Baja Inversión"
    elif 101000000 <= inv <= 500000000:
        grupo_inv = "Media Inversión"
    else:
        grupo_inv = "Alta Inversión"
    
    num_formatos = row['Num_Formatos']
    if 1 <= num_formatos <= 3:
        grupo_format = "Poca Variedad"
    elif 4 <= num_formatos <= 10:
        grupo_format = "Variedad Moderada"
    else:
        grupo_format = "Gran Variedad"
    
    num_plataformas = row['Num_Plataformas']
    if 1 <= num_plataformas <= 2:
        grupo_plat = "Cobertura Limitada"
    elif 3 <= num_plataformas <= 4:
        grupo_plat = "Cobertura Moderada"
    else:
        grupo_plat = "Cobertura Amplia"
    
    return f"{grupo_inv} – {grupo_format} – {grupo_plat}"

# Aplicar la función para asignar grupos
campaign_groups['Grupo'] = campaign_groups.apply(asignar_grupo, axis=1)

# =============================================================================
# 8B. Cálculo de Estadísticos Robustos por Grupo
# =============================================================================
# Calcular la mediana del final_score para cada campaña
# (final_score ya incluye el factor de impresiones)
campaign_summary = df.groupby('CAMPAIGN')['final_score'].median().reset_index()
campaign_summary.rename(columns={'final_score': 'median_score'}, inplace=True)

# Fusionar con la información de grupos
campaign_summary = pd.merge(campaign_summary, campaign_groups[['CAMPAIGN', 'Grupo']], on='CAMPAIGN', how='left')

# Calcular estadísticos robustos por grupo:
# Agrupar por 'Grupo' para obtener la mediana de las medianas
group_stats = campaign_summary.groupby('Grupo')['median_score'].agg(group_median='median').reset_index()

# Función para calcular el MAD robusto
def compute_mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))

# Calcular el MAD para cada grupo
group_mad = campaign_summary.groupby('Grupo')['median_score'].apply(compute_mad).reset_index()
group_mad.rename(columns={'median_score': 'group_mad'}, inplace=True)

# Fusionar los estadísticos de grupo con el resumen de campañas
campaign_summary = pd.merge(campaign_summary, group_stats, on='Grupo', how='left')
campaign_summary = pd.merge(campaign_summary, group_mad, on='Grupo', how='left')

# Calcular el z-score robusto para cada campaña dentro de su grupo
epsilon = 1e-7
campaign_summary['z_score'] = (campaign_summary['median_score'] - campaign_summary['group_median']) / (campaign_summary['group_mad'] + epsilon)

# =============================================================================
# 9. Exportación de Resultados a Excel
# =============================================================================
# Definir columnas para Hoja 1 (Assets); se incluyen los _rank_raw para referencia
base_columns = [
    'MONTH', 'PLATFORM', 'CREATIVE_ID', 'CREATIVE_NAME', 'CAMPAIGN', 'BRAND',
    'STAGE', 'AUDIENCE', 'FORMAT', 'CATEGORY', 'PURCHASE_TYPE',
    'IMPRESSIONS', 'VIDEO_VIEWS', 'COMPLETE_VIEWS', 'CLICS', 'COMMENTS',
    'INTERACTIONS', 'SHARES', 'REACH', 'MEDIA_SPEND', 'CPM', 'VTR',
    'CVTR', 'CTR', 'ER', 'VIEWABILITY'
]
result_columns = base_columns + [
    'Quality_Impressions', 'QCPM_calculated',
    'QCPM_ratio_norm', 'VIEWABILITY_ratio_norm', 'CVTR_ratio_norm', 'CTR_ratio_norm', 'ER_ratio_norm',
    'QCPM_rank_raw', 'Quality_Impressions_rank_raw', 'VIEWABILITY_rank_raw',
    'CVTR_rank_raw', 'CTR_rank_raw', 'ER_rank_raw',
    'QCPM_rank_norm', 'Quality_Impressions_rank_norm', 'VIEWABILITY_rank_norm',
    'CVTR_rank_norm', 'CTR_rank_norm', 'ER_rank_norm',
    'final_score', 'performance_index'
]
result_df = df[result_columns]

# Exportar a Excel con dos hojas:
with pd.ExcelWriter("final_results.xlsx", engine='openpyxl') as writer:
    result_df.to_excel(writer, sheet_name='Resultados por Asset', index=False)
    campaign_summary.to_excel(writer, sheet_name='Resumen de Campañas', index=False)

print("El archivo final ha sido guardado como 'final_results.xlsx'")
files.download("final_results.xlsx")
