"""
Clasificador Jerárquico con Bag of Visual Words especializado por grupo.

Este módulo implementa un clasificador de dos niveles donde:
- Nivel 1: Clasifica imágenes en grupos (meta-clases) usando un vocabulario visual global
- Nivel 2: Cada grupo tiene su propio vocabulario visual especializado y clasificador

Los hiperparámetros están optimizados para el dataset de comida india:
- Kernel: sigmoid
- C: 1.5
- K (vocabulario global): 2000

OPTIMIZACIONES:
- El K-means global se CARGA desde un pickle pre-entrenado (no se reentrena)
- Los K-means de los grupos se entrenan con MiniBatchKMeans
- Nivel 2 puede usar SVM o XGBoost (configurable)

Autor: [Tu nombre]
Fecha: [Fecha]
"""

import numpy as np
import pickle
import os
import json
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.base import clone
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from collections import defaultdict

# Para barras de progreso
USE_TQDM = True
try:
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False
    print("Advertencia: tqdm no instalado. No se mostrarán barras de progreso.")

# Intentar importar XGBoost
XGBOOST_DISPONIBLE = True
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBOOST_DISPONIBLE = False
    print("Advertencia: XGBoost no instalado. Solo se podrá usar SVM.")
    print("Para instalar XGBoost: pip install xgboost")


# =============================================================================
# CONFIGURACIÓN GLOBAL DE HIPERPARÁMETROS
# =============================================================================

# Hiperparámetros del SVM (optimizados para tu dataset)
KERNEL = 'sigmoid'
C_VALUE = 1.5
K_GLOBAL = 2000
K_POR_GRUPO = 500
N_GRUPOS_DEFAULT = 2
RANDOM_STATE = 42

# Configuración de MiniBatchKMeans (solo para los grupos, no para el global)
BATCH_SIZE = 10000
MAX_ITER = 100

# =============================================================================
# FLAG PARA ELEGIR CLASIFICADOR DE NIVEL 2
# =============================================================================
# None = preguntar, True = usar XGBoost, False = usar SVM

USE_XGBOOST = None

# Hiperparámetros de XGBoost
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'verbosity': 0
}


# =============================================================================
# FUNCIONES PARA CREAR CLASIFICADORES
# =============================================================================

def crear_svm():
    """Crea un SVM con los hiperparámetros óptimos."""
    return SVC(kernel=KERNEL, C=C_VALUE, random_state=RANDOM_STATE)


def crear_xgboost():
    """Crea un XGBClassifier con hiperparámetros razonables."""
    if not XGBOOST_DISPONIBLE:
        raise ImportError("XGBoost no está instalado. Instálalo con: pip install xgboost")
    return XGBClassifier(**XGBOOST_PARAMS)


def crear_clasificador_nivel2(usar_xgboost=False):
    """Crea el clasificador apropiado para el nivel 2."""
    if usar_xgboost:
        print(f"    → Usando XGBoost para este grupo")
        return crear_xgboost()
    else:
        print(f"    → Usando SVM (kernel={KERNEL}, C={C_VALUE}) para este grupo")
        return crear_svm()


def preguntar_usar_xgboost():
    """Pregunta al usuario si quiere usar XGBoost para el nivel 2."""
    if not XGBOOST_DISPONIBLE:
        print("\n⚠ XGBoost no está instalado. Se usará SVM para el nivel 2.")
        return False
    
    print("\n" + "="*60)
    print("SELECCIÓN DE CLASIFICADOR PARA NIVEL 2")
    print("="*60)
    print("\nOpciones para los clasificadores de los subgrupos (nivel 2):")
    print("  [1] SVM (kernel sigmoid, C=1.5) - Tus parámetros optimizados")
    print("  [2] XGBoost - Puede capturar relaciones no lineales más complejas")
    print("\nNota: El clasificador de nivel 1 (grupos) siempre usa SVM.")
    
    while True:
        respuesta = input("\n¿Qué clasificador quieres usar para el nivel 2? [1/2]: ").strip()
        if respuesta == '1':
            print("→ Se usará SVM para los clasificadores de nivel 2")
            return False
        elif respuesta == '2':
            print("→ Se usará XGBoost para los clasificadores de nivel 2")
            return True
        else:
            print("Por favor, introduce 1 o 2.")


# =============================================================================
# CLASE 1: Pipeline de Bag of Visual Words
# =============================================================================

class PipelineBoVW:
    """
    Pipeline completo de Bag of Visual Words.
    
    Puede funcionar de dos formas:
    1. Entrenar un nuevo vocabulario con MiniBatchKMeans
    2. Cargar un vocabulario pre-entrenado desde un pickle
    """
    
    def __init__(self, k=K_GLOBAL, random_state=RANDOM_STATE):
        """
        Inicializa el pipeline.
        
        Args:
            k: Número de palabras visuales (clusters).
            random_state: Semilla para reproducibilidad.
        """
        self.k = k
        self.random_state = random_state
        self.kmeans = None
        self.vocabulario_entrenado = False
    
    def cargar_vocabulario(self, kmeans_entrenado):
        """
        Carga un K-means ya entrenado (desde un pickle o variable).
        
        Esto es útil cuando ya tienes un vocabulario visual optimizado
        y no quieres volver a entrenarlo.
        
        Args:
            kmeans_entrenado: Objeto KMeans o MiniBatchKMeans ya entrenado.
                             Puede ser cargado desde un pickle.
        
        Returns:
            self: Para encadenamiento de métodos.
        """
        self.kmeans = kmeans_entrenado
        self.k = kmeans_entrenado.n_clusters
        self.vocabulario_entrenado = True
        print(f"    Vocabulario cargado: K={self.k}")
        return self
    
    def cargar_vocabulario_desde_pickle(self, filepath):
        """
        Carga un K-means desde un archivo pickle.
        
        Args:
            filepath: Ruta al archivo pickle con el K-means entrenado.
        
        Returns:
            self: Para encadenamiento de métodos.
        """
        print(f"    Cargando vocabulario desde: {filepath}")
        with open(filepath, 'rb') as f:
            kmeans_entrenado = pickle.load(f)
        return self.cargar_vocabulario(kmeans_entrenado)
    
    def entrenar_vocabulario(self, descriptores_por_imagen):
        """
        Entrena el vocabulario visual usando MiniBatchKMeans.
        
        Args:
            descriptores_por_imagen: Lista de arrays numpy con los descriptores.
        
        Returns:
            self: Para encadenamiento de métodos.
        """
        print(f"    Concatenando descriptores de {len(descriptores_por_imagen)} imágenes...")
        todos_descriptores = np.vstack(descriptores_por_imagen)
        print(f"    Total de descriptores: {len(todos_descriptores)}")
        print(f"    Dimensión de cada descriptor: {todos_descriptores.shape[1]}")
        print(f"    Entrenando MiniBatchKMeans con K={self.k} (batch_size={BATCH_SIZE})...")
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            batch_size=BATCH_SIZE,
            max_iter=MAX_ITER,
            verbose=1,
            n_init=3
        )
        self.kmeans.fit(todos_descriptores)
        self.vocabulario_entrenado = True
        print(f"    MiniBatchKMeans entrenado. Inercia: {self.kmeans.inertia_:.2f}")
        
        return self
    
    def imagen_a_histograma(self, descriptores_imagen):
        """
        Convierte los descriptores de una imagen en un histograma normalizado.
        
        Args:
            descriptores_imagen: Array de shape (n_keypoints, dim_descriptor)
        
        Returns:
            Histograma normalizado de shape (k,)
        """
        if not self.vocabulario_entrenado:
            raise ValueError("Primero debes entrenar o cargar el vocabulario")
        
        palabras = self.kmeans.predict(descriptores_imagen)
        histograma = np.bincount(palabras, minlength=self.k).astype(float)
        
        norma = np.linalg.norm(histograma)
        if norma > 0:
            histograma = histograma / norma
        
        return histograma
    
    def imagenes_a_histogramas(self, lista_descriptores, mostrar_progreso=True):
        """
        Convierte una lista de descriptores en una matriz de histogramas.
        
        Args:
            lista_descriptores: Lista de arrays de descriptores.
            mostrar_progreso: Si True, muestra barra de progreso.
        
        Returns:
            Matriz de shape (n_imagenes, k)
        """
        histogramas = []
        
        if mostrar_progreso and USE_TQDM:
            iterador = tqdm(lista_descriptores, desc="    Construyendo histogramas")
        else:
            iterador = lista_descriptores
        
        for desc in iterador:
            hist = self.imagen_a_histograma(desc)
            histogramas.append(hist)
        
        return np.array(histogramas)


# =============================================================================
# CLASE 2: Clasificador Jerárquico Completo
# =============================================================================

class ClasificadorJerarquicoBoVW:
    """
    Clasificador jerárquico de dos niveles con pipelines BoVW independientes.
    
    Arquitectura:
    - Nivel 1: Pipeline BoVW global (K-means cargado desde pickle) + SVM
    - Nivel 2: Pipeline BoVW especializado (K-means entrenado) + clasificador (SVM/XGBoost)
    """
    
    def __init__(self, n_grupos=N_GRUPOS_DEFAULT, k_global=K_GLOBAL, k_por_grupo=K_POR_GRUPO,
                 usar_xgboost_nivel2=False):
        """
        Inicializa el clasificador jerárquico.
        
        Args:
            n_grupos: Número de meta-clases para el nivel 1.
            k_global: Tamaño del vocabulario global (para referencia).
            k_por_grupo: Tamaño del vocabulario para cada grupo (int o dict).
            usar_xgboost_nivel2: Si True, usa XGBoost para nivel 2. Si False, usa SVM.
        """
        self.n_grupos = n_grupos
        self.k_global = k_global
        self.usar_xgboost_nivel2 = usar_xgboost_nivel2
        
        # Permitir K diferente por grupo
        if isinstance(k_por_grupo, int):
            self.k_por_grupo = {g: k_por_grupo for g in range(n_grupos)}
        else:
            self.k_por_grupo = k_por_grupo
        
        # Componentes del nivel 1
        self.pipeline_global = None
        self.clasificador_nivel1 = None
        
        # Componentes del nivel 2
        self.pipelines_grupo = {}
        self.clasificadores_grupo = {}
        self.label_encoders_grupo = {}
        
        # Mapeos de clases
        self.clase_a_grupo = {}
        self.clases_por_grupo = {}
        self.clases_unicas = None
        
        # Datos de entrenamiento
        self._X_global_train = None
        self._y_train = None
        self._y_grupos_train = None
    
    def definir_grupos_desde_confusion(self, cm, clases, mostrar_dendrograma=True):
        """
        Define los grupos automáticamente usando clustering jerárquico
        sobre la matriz de confusión.
        """
        # Simetrizar la matriz
        cm_sym = cm + cm.T
        
        # Normalizar
        row_sums = cm_sym.sum(axis=1, keepdims=True)
        cm_norm = cm_sym / (row_sums + 1e-10)
        
        # Convertir similitud a distancia
        np.fill_diagonal(cm_norm, 0)
        max_val = cm_norm.max()
        dist_matrix = max_val - cm_norm
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        
        # Clustering jerárquico
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method='ward')
        
        # Mostrar dendrograma
        if mostrar_dendrograma:
            plt.figure(figsize=(14, 6))
            dendrogram(Z, labels=list(clases), leaf_rotation=45)
            plt.xlabel('Clases')
            plt.ylabel('Distancia')
            plt.title('Dendrograma de similitud entre clases\n(basado en confusiones)')
            plt.tight_layout()
            plt.show()
        
        # Cortar el dendrograma
        grupos = fcluster(Z, self.n_grupos, criterion='maxclust')
        
        # Crear mapeos
        self.clases_unicas = np.array(clases)
        for clase, grupo in zip(clases, grupos):
            self.clase_a_grupo[clase] = grupo - 1
        
        for g in range(self.n_grupos):
            self.clases_por_grupo[g] = [c for c in clases if self.clase_a_grupo[c] == g]
        
        # Mostrar grupos
        print("\n" + "="*60)
        print("GRUPOS DEFINIDOS AUTOMÁTICAMENTE")
        print("="*60)
        for g in range(self.n_grupos):
            print(f"  Grupo {g}: {self.clases_por_grupo[g]}")
        
        return self
    
    def definir_grupos_manual(self, clase_a_grupo):
        """Define los grupos manualmente."""
        self.clase_a_grupo = clase_a_grupo.copy()
        self.clases_unicas = np.array(sorted(clase_a_grupo.keys()))
        
        for g in range(self.n_grupos):
            self.clases_por_grupo[g] = [c for c, grupo in clase_a_grupo.items() if grupo == g]
        
        print("\n" + "="*60)
        print("GRUPOS DEFINIDOS MANUALMENTE")
        print("="*60)
        for g in range(self.n_grupos):
            print(f"  Grupo {g}: {self.clases_por_grupo[g]}")
        
        return self
    
    def _preparar_datos_desde_pickle(self, datos_pickle):
        """Convierte la estructura del pickle a listas paralelas."""
        lista_descriptores = []
        lista_etiquetas = []
        lista_paths = []
        
        for clase in sorted(datos_pickle.keys()):
            imagenes = datos_pickle[clase]
            for path in sorted(imagenes.keys()):
                descriptores = imagenes[path]
                lista_descriptores.append(descriptores)
                lista_etiquetas.append(clase)
                lista_paths.append(path)
        
        return lista_descriptores, np.array(lista_etiquetas), lista_paths
    
    def fit(self, datos_train_pickle, kmeans_global_pickle_path, clasificador_nivel1=None):
        """
        Entrena el clasificador jerárquico completo.
        
        Args:
            datos_train_pickle: Diccionario del pickle de entrenamiento
            kmeans_global_pickle_path: Ruta al pickle con el K-means global ya entrenado
            clasificador_nivel1: Clasificador para nivel 1 (default: SVM)
        
        Returns:
            self
        """
        if not self.clase_a_grupo:
            raise ValueError("Primero debes definir los grupos")
        
        # Preparar datos
        print("\n" + "="*60)
        print("PREPARANDO DATOS DE ENTRENAMIENTO")
        print("="*60)
        descriptores_train, y_train, paths_train = self._preparar_datos_desde_pickle(datos_train_pickle)
        n_imagenes = len(descriptores_train)
        print(f"Total de imágenes de entrenamiento: {n_imagenes}")
        print(f"Número de clases: {len(np.unique(y_train))}")
        
        tipo_clf_nivel2 = "XGBoost" if self.usar_xgboost_nivel2 else "SVM"
        print(f"Clasificador nivel 2: {tipo_clf_nivel2}")
        
        # Convertir etiquetas a grupos
        y_grupos = np.array([self.clase_a_grupo[c] for c in y_train])
        
        # ================================================================
        # NIVEL 1: Pipeline global (CARGA K-MEANS DESDE PICKLE)
        # ================================================================
        print("\n" + "="*60)
        print("NIVEL 1: CARGANDO PIPELINE GLOBAL DESDE PICKLE")
        print("="*60)
        print(f"  Configuración: SVM (kernel={KERNEL}, C={C_VALUE})")
        
        # CARGAR el K-means global desde el pickle (NO entrenar)
        self.pipeline_global = PipelineBoVW(k=self.k_global)
        self.pipeline_global.cargar_vocabulario_desde_pickle(kmeans_global_pickle_path)
        
        # Construir histogramas globales
        print("  Construyendo histogramas globales...")
        X_global = self.pipeline_global.imagenes_a_histogramas(descriptores_train)
        print(f"  Shape de histogramas globales: {X_global.shape}")
        
        # Entrenar clasificador de nivel 1 (siempre SVM)
        print(f"  Entrenando clasificador de nivel 1 (predice {self.n_grupos} grupos)...")
        self.clasificador_nivel1 = clasificador_nivel1 or crear_svm()
        self.clasificador_nivel1.fit(X_global, y_grupos)
        
        # Evaluar accuracy del nivel 1
        y_grupos_pred_train = self.clasificador_nivel1.predict(X_global)
        acc_nivel1_train = accuracy_score(y_grupos, y_grupos_pred_train)
        print(f"  Accuracy nivel 1 (train): {acc_nivel1_train:.4f}")
        
        # Guardar para diagnóstico
        self._X_global_train = X_global
        self._y_train = y_train
        self._y_grupos_train = y_grupos
        
        # ================================================================
        # NIVEL 2: Pipelines especializados (ENTRENA K-MEANS POR GRUPO)
        # ================================================================
        print("\n" + "="*60)
        print(f"NIVEL 2: ENTRENANDO PIPELINES ESPECIALIZADOS ({tipo_clf_nivel2})")
        print("="*60)
        
        for grupo in range(self.n_grupos):
            clases_en_grupo = self.clases_por_grupo[grupo]
            print(f"\n  {'─'*56}")
            print(f"  GRUPO {grupo}: {clases_en_grupo}")
            print(f"  {'─'*56}")
            
            # Si el grupo tiene solo una clase, no necesitamos clasificador
            if len(clases_en_grupo) <= 1:
                print(f"    → Solo una clase, no se necesita clasificador")
                self.pipelines_grupo[grupo] = None
                self.clasificadores_grupo[grupo] = None
                self.label_encoders_grupo[grupo] = None
                continue
            
            # Filtrar imágenes del grupo
            mask_grupo = np.array([self.clase_a_grupo[c] == grupo for c in y_train])
            descriptores_grupo = [descriptores_train[i] for i in range(n_imagenes) if mask_grupo[i]]
            y_grupo = y_train[mask_grupo]
            
            print(f"    Imágenes en este grupo: {len(descriptores_grupo)}")
            print(f"    Distribución de clases:")
            for clase in clases_en_grupo:
                n_clase = (y_grupo == clase).sum()
                print(f"      - {clase}: {n_clase} imágenes")
            
            # ENTRENAR vocabulario específico del grupo (MiniBatchKMeans)
            k_grupo = self.k_por_grupo.get(grupo, K_POR_GRUPO)
            print(f"    Entrenando vocabulario específico (K={k_grupo})...")
            self.pipelines_grupo[grupo] = PipelineBoVW(k=k_grupo)
            self.pipelines_grupo[grupo].entrenar_vocabulario(descriptores_grupo)
            
            # Construir histogramas específicos
            X_grupo = self.pipelines_grupo[grupo].imagenes_a_histogramas(descriptores_grupo)
            print(f"    Shape de histogramas del grupo: {X_grupo.shape}")
            
            # Preparar etiquetas
            if self.usar_xgboost_nivel2:
                le = LabelEncoder()
                y_grupo_encoded = le.fit_transform(y_grupo)
                self.label_encoders_grupo[grupo] = le
            else:
                y_grupo_encoded = y_grupo
                self.label_encoders_grupo[grupo] = None
            
            # Crear y entrenar clasificador del grupo
            clf = crear_clasificador_nivel2(usar_xgboost=self.usar_xgboost_nivel2)
            clf.fit(X_grupo, y_grupo_encoded)
            self.clasificadores_grupo[grupo] = clf
            
            # Evaluar accuracy del grupo
            y_grupo_pred = clf.predict(X_grupo)
            if self.usar_xgboost_nivel2:
                y_grupo_pred = self.label_encoders_grupo[grupo].inverse_transform(y_grupo_pred)
            acc_grupo_train = accuracy_score(y_grupo, y_grupo_pred)
            print(f"    Accuracy grupo {grupo} (train): {acc_grupo_train:.4f}")
        
        print("\n" + "="*60)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*60)
        
        return self
    
    def predict(self, datos_test_pickle):
        """Predice las clases para nuevas imágenes."""
        descriptores_test, y_test, paths_test = self._preparar_datos_desde_pickle(datos_test_pickle)
        n_imagenes = len(descriptores_test)
        
        # Construir histogramas globales
        print("  Construyendo histogramas globales para test...")
        X_global = self.pipeline_global.imagenes_a_histogramas(descriptores_test, mostrar_progreso=False)
        
        # Predecir grupos con nivel 1
        grupos_pred = self.clasificador_nivel1.predict(X_global)
        
        # Predecir clases finales con nivel 2
        predicciones = []
        
        print("  Prediciendo clases finales...")
        if USE_TQDM:
            iterador = tqdm(range(n_imagenes), desc="  Predicción nivel 2")
        else:
            iterador = range(n_imagenes)
        
        for i in iterador:
            grupo = grupos_pred[i]
            
            if self.clasificadores_grupo[grupo] is None:
                predicciones.append(self.clases_por_grupo[grupo][0])
            else:
                hist_grupo = self.pipelines_grupo[grupo].imagen_a_histograma(descriptores_test[i])
                pred = self.clasificadores_grupo[grupo].predict(hist_grupo.reshape(1, -1))[0]
                
                if self.usar_xgboost_nivel2 and self.label_encoders_grupo[grupo] is not None:
                    pred = self.label_encoders_grupo[grupo].inverse_transform([pred])[0]
                
                predicciones.append(pred)
        
        return np.array(predicciones), y_test, paths_test
    
    def evaluar(self, datos_test_pickle, mostrar_reporte=True):
        """Evalúa el clasificador jerárquico con métricas detalladas."""
        descriptores_test, y_test, paths_test = self._preparar_datos_desde_pickle(datos_test_pickle)
        n_imagenes = len(descriptores_test)
        
        # Convertir etiquetas reales a grupos
        y_grupos_real = np.array([self.clase_a_grupo[c] for c in y_test])
        
        # Predecir grupos (nivel 1)
        print("  Evaluando nivel 1...")
        X_global = self.pipeline_global.imagenes_a_histogramas(descriptores_test, mostrar_progreso=False)
        grupos_pred = self.clasificador_nivel1.predict(X_global)
        
        # Predecir clases finales (nivel 2)
        print("  Evaluando nivel 2...")
        y_pred, _, _ = self.predict(datos_test_pickle)
        
        # Calcular métricas
        acc_nivel1 = accuracy_score(y_grupos_real, grupos_pred)
        
        mask_nivel1_ok = grupos_pred == y_grupos_real
        if mask_nivel1_ok.sum() > 0:
            acc_nivel2_cond = accuracy_score(y_test[mask_nivel1_ok], y_pred[mask_nivel1_ok])
        else:
            acc_nivel2_cond = 0.0
        
        acc_final = accuracy_score(y_test, y_pred)
        
        tipo_clf = "XGBoost" if self.usar_xgboost_nivel2 else "SVM"
        
        # Imprimir resultados
        print("\n" + "="*60)
        print(f"EVALUACIÓN DEL CLASIFICADOR JERÁRQUICO (Nivel 2: {tipo_clf})")
        print("="*60)
        print(f"\n  Accuracy nivel 1 (grupos):              {acc_nivel1:.4f} ({acc_nivel1*100:.2f}%)")
        print(f"  Accuracy nivel 2 (dado nivel 1 OK):     {acc_nivel2_cond:.4f} ({acc_nivel2_cond*100:.2f}%)")
        print(f"  Producto teórico (nivel1 × nivel2):     {acc_nivel1 * acc_nivel2_cond:.4f} ({acc_nivel1 * acc_nivel2_cond*100:.2f}%)")
        print(f"  Accuracy final real:                    {acc_final:.4f} ({acc_final*100:.2f}%)")
        
        # Desglose por grupo
        print(f"\n  Desglose por grupo (nivel 1):")
        for g in range(self.n_grupos):
            mask_real = y_grupos_real == g
            mask_pred = grupos_pred == g
            
            n_real = mask_real.sum()
            n_pred = mask_pred.sum()
            n_correcto = (mask_real & mask_pred).sum()
            
            recall = n_correcto / n_real if n_real > 0 else 0
            precision = n_correcto / n_pred if n_pred > 0 else 0
            
            clases_str = ', '.join(self.clases_por_grupo[g][:3])
            if len(self.clases_por_grupo[g]) > 3:
                clases_str += f", ... ({len(self.clases_por_grupo[g])} clases)"
            
            print(f"    Grupo {g} [{clases_str}]:")
            print(f"      real={n_real}, pred={n_pred}, precision={precision:.3f}, recall={recall:.3f}")
        
        # Classification report
        if mostrar_reporte:
            print(f"\n  Classification Report (clases individuales):")
            print("  " + "-"*56)
            print(classification_report(y_test, y_pred, zero_division=0))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=self.clases_unicas)
        
        return {
            'accuracy_nivel1': acc_nivel1,
            'accuracy_nivel2_condicional': acc_nivel2_cond,
            'accuracy_final': acc_final,
            'y_test': y_test,
            'y_pred': y_pred,
            'grupos_pred': grupos_pred,
            'grupos_real': y_grupos_real,
            'confusion_matrix': cm,
            'clases': self.clases_unicas,
            'tipo_clasificador_nivel2': tipo_clf
        }
    
    def guardar(self, filepath):
        """Guarda el modelo completo."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Modelo guardado en: {filepath}")
    
    @staticmethod
    def cargar(filepath):
        """Carga un modelo previamente guardado."""
        with open(filepath, 'rb') as f:
            modelo = pickle.load(f)
        print(f"Modelo cargado desde: {filepath}")
        return modelo


# =============================================================================
# FUNCIÓN: Entrenar clasificador plano (baseline)
# =============================================================================

def entrenar_clasificador_plano(datos_train_pickle, datos_test_pickle, kmeans_pickle_path, k=K_GLOBAL):
    """
    Entrena un clasificador plano (baseline) para comparación.
    
    Carga el K-means desde un pickle en lugar de entrenarlo.
    """
    print("\n" + "="*60)
    print("ENTRENANDO CLASIFICADOR PLANO (BASELINE)")
    print("="*60)
    print(f"  Configuración: K={k}, kernel={KERNEL}, C={C_VALUE}")
    
    def preparar_datos(datos):
        lista_desc, lista_etiq, lista_paths = [], [], []
        for clase in sorted(datos.keys()):
            imagenes = datos[clase]
            for path in sorted(imagenes.keys()):
                desc = imagenes[path]
                lista_desc.append(desc)
                lista_etiq.append(clase)
                lista_paths.append(path)
        return lista_desc, np.array(lista_etiq), lista_paths
    
    desc_train, y_train, _ = preparar_datos(datos_train_pickle)
    desc_test, y_test, _ = preparar_datos(datos_test_pickle)
    
    print(f"  Imágenes train: {len(desc_train)}")
    print(f"  Imágenes test: {len(desc_test)}")
    print(f"  Número de clases: {len(np.unique(y_train))}")
    
    # CARGAR K-means desde pickle (no entrenar)
    pipeline = PipelineBoVW(k=k)
    pipeline.cargar_vocabulario_desde_pickle(kmeans_pickle_path)
    
    X_train = pipeline.imagenes_a_histogramas(desc_train)
    X_test = pipeline.imagenes_a_histogramas(desc_test)
    
    # Entrenar SVM
    print(f"  Entrenando SVM (kernel={KERNEL}, C={C_VALUE})...")
    clf = crear_svm()
    clf.fit(X_train, y_train)
    
    # Evaluar
    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n  Accuracy (train): {acc_train:.4f} ({acc_train*100:.2f}%)")
    print(f"  Accuracy (test):  {acc:.4f} ({acc*100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return {
        'accuracy_train': acc_train,
        'accuracy': acc,
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


# =============================================================================
# FUNCIÓN: Visualizar comparación
# =============================================================================

def visualizar_comparacion(resultado_plano, resultado_jerarquico, clases):
    """Visualiza la comparación de matrices de confusión."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    im1 = axes[0].imshow(resultado_plano['confusion_matrix'], cmap='Blues')
    axes[0].set_title(f"Clasificador Plano\nAccuracy: {resultado_plano['accuracy']:.4f}")
    axes[0].set_xlabel('Predicción')
    axes[0].set_ylabel('Real')
    plt.colorbar(im1, ax=axes[0])
    
    tipo_clf = resultado_jerarquico.get('tipo_clasificador_nivel2', 'SVM')
    im2 = axes[1].imshow(resultado_jerarquico['confusion_matrix'], cmap='Blues')
    axes[1].set_title(f"Clasificador Jerárquico ({tipo_clf})\nAccuracy: {resultado_jerarquico['accuracy_final']:.4f}")
    axes[1].set_xlabel('Predicción')
    axes[1].set_ylabel('Real')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURACIÓN - Ajusta estos paths
    # =========================================================================
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Paths a tus archivos - AJUSTA ESTOS PATHS
    PATH_TRAIN_PICKLE = os.path.join(BASE_DIR, 'train_data_sift,splitted.pickle')
    PATH_TEST_PICKLE = os.path.join(BASE_DIR, 'test_data_sift,splitted.pickle')
    PATH_STATS_JSON = os.path.join(BASE_DIR, 'STATS_IMPORTANTES', 'estadisticas_sigmoide_cMillor.json')
    
    # PATH AL KMEANS GLOBAL YA ENTRENADO
    PATH_KMEANS_GLOBAL = os.path.join(BASE_DIR, 'kmeans_sift,splitted_2000.pickle')
    
    N_GRUPOS = 4
    K_GRUPO = 500
    
    # =========================================================================
    # PASO 1: Cargar datos
    # =========================================================================
    
    print("\n" + "="*60)
    print("CARGANDO DATOS")
    print("="*60)
    
    print(f"  Cargando pickle de train: {PATH_TRAIN_PICKLE}")
    with open(PATH_TRAIN_PICKLE, 'rb') as f:
        datos_train = pickle.load(f)
    print(f"  → {len(datos_train)} clases cargadas")
    
    print(f"  Cargando pickle de test: {PATH_TEST_PICKLE}")
    with open(PATH_TEST_PICKLE, 'rb') as f:
        datos_test = pickle.load(f)
    print(f"  → {len(datos_test)} clases cargadas")
    
    total_train = sum(len(imgs) for imgs in datos_train.values())
    total_test = sum(len(imgs) for imgs in datos_test.values())
    print(f"  Total imágenes train: {total_train}")
    print(f"  Total imágenes test: {total_test}")
    
    # Verificar que el archivo del K-means existe
    if not os.path.exists(PATH_KMEANS_GLOBAL):
        print(f"\n❌ ERROR: No se encontró el archivo del K-means global:")
        print(f"   {PATH_KMEANS_GLOBAL}")
        print(f"   Asegúrate de que el path es correcto.")
        exit(1)
    else:
        print(f"\n  ✓ K-means global encontrado: {PATH_KMEANS_GLOBAL}")
    
    # =========================================================================
    # PASO 2: Cargar matriz de confusión
    # =========================================================================
    
    print("\n" + "="*60)
    print("CARGANDO MATRIZ DE CONFUSIÓN PREVIA")
    print("="*60)
    
    print(f"  Cargando: {PATH_STATS_JSON}")
    with open(PATH_STATS_JSON, 'r') as f:
        stats = json.load(f)
    
    cm = np.array(stats['sift,splitted']['2000']['confusion_matrix'])
    clases = sorted(datos_train.keys())
    
    print(f"  Matriz de confusión: shape {cm.shape}")
    print(f"  Clases: {clases}")
    print(f"  F1-score del mejor clasificador previo: {stats['sift,splitted']['2000']['f1']:.4f}")
    
    # =========================================================================
    # PASO 3: Entrenar clasificador plano (baseline)
    # =========================================================================
    
    resultado_plano = entrenar_clasificador_plano(
        datos_train, 
        datos_test, 
        kmeans_pickle_path=PATH_KMEANS_GLOBAL,  # Cargar K-means desde pickle
        k=K_GLOBAL
    )
    
    # =========================================================================
    # PASO 4: PREGUNTAR SI USAR XGBOOST
    # =========================================================================
    
    if USE_XGBOOST is None:
        usar_xgboost = preguntar_usar_xgboost()
    else:
        usar_xgboost = USE_XGBOOST
        tipo = "XGBoost" if usar_xgboost else "SVM"
        print(f"\n→ Usando {tipo} para nivel 2 (configurado en USE_XGBOOST)")
    
    # =========================================================================
    # PASO 5: Crear y entrenar clasificador jerárquico
    # =========================================================================
    
    clf_jerarquico = ClasificadorJerarquicoBoVW(
        n_grupos=N_GRUPOS,
        k_global=K_GLOBAL,
        k_por_grupo=K_GRUPO,
        usar_xgboost_nivel2=usar_xgboost
    )
    
    # Definir grupos desde la matriz de confusión
    clf_jerarquico.definir_grupos_desde_confusion(cm, clases, mostrar_dendrograma=True)
    
    # Entrenar (pasando el path al K-means global)
    clf_jerarquico.fit(
        datos_train,
        kmeans_global_pickle_path=PATH_KMEANS_GLOBAL,  # Cargar K-means desde pickle
        clasificador_nivel1=crear_svm()
    )
    
    # =========================================================================
    # PASO 6: Evaluar y comparar
    # =========================================================================
    
    resultado_jerarquico = clf_jerarquico.evaluar(datos_test)
    
    # =========================================================================
    # PASO 7: Comparación final
    # =========================================================================
    
    print("\n" + "="*60)
    print("COMPARACIÓN FINAL")
    print("="*60)
    
    acc_plano = resultado_plano['accuracy']
    acc_jerarquico = resultado_jerarquico['accuracy_final']
    tipo_clf = resultado_jerarquico['tipo_clasificador_nivel2']
    
    print(f"\n  Clasificador PLANO (baseline):              {acc_plano:.4f} ({acc_plano*100:.2f}%)")
    print(f"  Clasificador JERÁRQUICO (nivel 2: {tipo_clf}):  {acc_jerarquico:.4f} ({acc_jerarquico*100:.2f}%)")
    
    diferencia = acc_jerarquico - acc_plano
    diferencia_pct = diferencia * 100
    
    if diferencia > 0:
        print(f"\n  ✓ El clasificador jerárquico MEJORA en {diferencia_pct:.2f}%")
    elif diferencia < 0:
        print(f"\n  ✗ El clasificador jerárquico EMPEORA en {abs(diferencia_pct):.2f}%")
    else:
        print(f"\n  = Ambos clasificadores tienen el mismo rendimiento")
    
    # Análisis adicional
    print(f"\n  Análisis del clasificador jerárquico:")
    print(f"    - Accuracy nivel 1 (grupos): {resultado_jerarquico['accuracy_nivel1']:.4f}")
    print(f"    - Accuracy nivel 2 (clases|grupo correcto): {resultado_jerarquico['accuracy_nivel2_condicional']:.4f}")
    
    if resultado_jerarquico['accuracy_nivel1'] < 0.9:
        print(f"\n  ⚠ El nivel 1 tiene accuracy < 90%. Considera:")
        print(f"    - Reducir el número de grupos")
        print(f"    - Revisar si los grupos tienen sentido semántico")
    
    if resultado_jerarquico['accuracy_nivel2_condicional'] < acc_plano:
        print(f"\n  ⚠ El nivel 2 no mejora respecto al plano. Considera:")
        print(f"    - Aumentar K por grupo")
        print(f"    - Probar el otro clasificador ({'SVM' if usar_xgboost else 'XGBoost'})")
    
    # =========================================================================
    # PASO 8: Visualizar (opcional)
    # =========================================================================
    
    # Descomenta para ver las matrices de confusión
    # visualizar_comparacion(resultado_plano, resultado_jerarquico, clases)
    
    # =========================================================================
    # PASO 9: Guardar el modelo (opcional)
    # =========================================================================
    
    # Descomenta si quieres guardar el modelo
    # nombre_archivo = f'clasificador_jerarquico_{tipo_clf.lower()}.pkl'
    # clf_jerarquico.guardar(nombre_archivo)
    
    print("\n" + "="*60)
    print("EJECUCIÓN COMPLETADA")
    print("="*60)