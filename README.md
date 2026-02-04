# Machine Learning - Indian Food Photo Detection -- Clasificador de comida india basado en Bag of Visual Words (BoVW) con SIFT y SVM.

## Descripción
Pipeline completo de clasificación de imágenes implementando el modelo BoVW: extrae características locales SIFT, construye vocabulario visual mediante K-Means, convierte imágenes a histogramas y clasifica con SVM (kernels RBF y Sigmoid).

## Estructura del Proyecto
Machine_Learning_Indianfood_PhotoDetect/
├── .github/
│ ├── Food Classification/ # Dataset: 20 categorías de comida
│ └── files/ # Scripts y modelos
│ ├── 00_main.py # Script principal del pipeline
│ ├── a_local_feature_extraction.py # Extracción SIFT
│ ├── b_vocabulary_extraction.py # Construcción de vocabulario (K-Means)
│ ├── c_convert_nontabular_to_tabular.py # Conversión a histogramas
│ ├── d_metric_visualization.py # Visualización (ROC, t-SNE, heatmaps)
│ ├── e_auxiliar_transformations.py # Transformaciones auxiliares
│ ├── ver_estadisticas.py # Análisis de resultados
│ ├── kmeans_sift_*.pickle # Modelos K-Means (k=10,500,1000,3000,3500,4000,4500,5000)
│ └── estadisticas_k2.json # Resultados de experimentos
├── Informe_ML_Grup11.pdf # Documentación técnica
├── presentacio final.pptx # Presentación
└── pruebas.py # Script de pruebas

## Dataset

**20 categorías:** Butter Naan, Chai, Chapati, Chole Bhature, Dal Makhani, Dhokla, Idli, Jalebi, Kaathi Rolls, Kadai Paneer, Kulfi, Masala Dosa, Momos, Paani Puri, Pakode, Pav Bhaji, Samosa, Burger, Fried Rice, Pizza.

Pipeline
1. **Preprocesamiento**: Redimensionado de imágenes (300x300)
2. **Extracción SIFT**: Detección de keypoints y descriptores locales
3. **Vocabulario Visual**: K-Means sobre descriptores (vocabulario de palabras visuales)
4. **Histogramas BoVW**: Representación tabular de imágenes
5. **Clasificación**: SVM con kernels RBF y Sigmoid
6. **Evaluación**: Métricas, ROC curves, t-SNE, heatmaps

## Uso

bash
cd Machine_Learning_Indianfood_PhotoDetect/.github/files
python 00_main.py

Configuración en 00_main.py:

methods: Método de extracción (ej: ['sift,splitted'])
k: Tamaño del vocabulario (ej: [2000])
kernel: Kernel SVM ('rbf' o 'sigmoid')


