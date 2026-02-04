import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import random
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import pickle

def visualitzar_progressio_scores(filepath, type, output_dir=None):
    """
    Genera una gràfica de l'evolució del F1-score, Accuracy test i Accuracy train segons K per a cada mètode.
    
    Args:
        filepath: Ruta al fitxer JSON amb les estadístiques
        output_dir: Directori on guardar la imatge (opcional, per defecte el mateix que el JSON)
    """
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    plt.figure(figsize=(14, 8))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

    for idx, (metode, resultats_k) in enumerate(data.items()):
        valors_k = sorted([k for k in resultats_k.keys()])
        
        f1_scores = [resultats_k[str(k)]['f1'] for k in valors_k]
        accuracy_test = [resultats_k[str(k)]['accuracy'] for k in valors_k]
        
        # Verificar si existe accuracy_train
        if 'accuracy_train' in resultats_k[str(valors_k[0])]:
            accuracy_train = [resultats_k[str(k)]['accuracy_train'] for k in valors_k]
        else:
            print(f"⚠️  '{metode}' no té accuracy_train. Executa primer el training amb el codi modificat.")
            accuracy_train = None
        
        color = colors[idx % len(colors)]
        
        # Línia 1: F1-score test (sòlida amb cercle)
        plt.plot(valors_k, f1_scores, 
                 color=color, linestyle='-', marker='o', markersize=5, linewidth=2,
                 label=f'{metode} - F1 (test)')
        
        # Línia 2: Accuracy test (discontínua amb quadrat)
        plt.plot(valors_k, accuracy_test, 
                 color=color, linestyle='--', marker='s', markersize=4, linewidth=1.5,
                 label=f'{metode} - Accuracy (test)')
        
        # Línia 3: Accuracy train (puntejada amb triangle)
        if accuracy_train:
            plt.plot(valors_k, accuracy_train, 
                     color=color, linestyle=':', marker='^', markersize=4, linewidth=1.5,
                     label=f'{metode} - Accuracy (train)')
    
    # Configurar títol i etiquetes
    nom_kernel = os.path.basename(filepath).replace('estadisticas_', '').replace('.json', '')
    plt.title(f"Evolució de mètriques segons K (Kernel: {nom_kernel})\n" + 
              "Gap entre train (···) i test (---) = OVERFITTING", 
              fontsize=13, fontweight='bold')
    plt.xlabel("Nombre de clusters (K)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    
    # Afegir llegenda fora del gràfic per no tapar les línies
    plt.legend(title="Mètode - Mètrica", loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    
    # Afegir graella per facilitar la lectura
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ajustar els límits de l'eix Y entre 0 i 1
    plt.ylim(0, 1)
    
    # Ajustar el layout perquè la llegenda no quedi tallada
    plt.tight_layout()
    
    # Generar el nom del fitxer de sortida
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    
    nom_sortida = f"f1_evolution_{nom_kernel}.png"
    path_sortida = os.path.join(output_dir, nom_sortida)
    
    # Guardar la figura amb alta resolució
    plt.savefig(path_sortida, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gràfica guardada a: {path_sortida}")
    return path_sortida


def visualitzar_tots_els_kernels(directori, type = 'simple'):
    """
    Genera gràfiques per a tots els fitxers d'estadístiques d'un directori.
    
    Args:
        directori: Ruta al directori que conté els JSON d'estadístiques
    """
    
    # Buscar tots els fitxers JSON d'estadístiques
    fitxers = [f for f in os.listdir(directori) if f.startswith('estadisticas_') and f.endswith('.json')]
    
    if not fitxers:
        print("No s'han trobat fitxers d'estadístiques al directori especificat.")
        return
    
    print(f"Trobats {len(fitxers)} fitxers d'estadístiques:")
    
    for fitxer in fitxers:
        filepath = os.path.join(directori, fitxer)
        print(f"\n  Processant: {fitxer}")
        visualitzar_progressio_scores(filepath, type)

def generar_heatmap_millors_k(directori, output_path=None):
    """
    Genera un mapa de calor mostrant la millor K per a cada combinació mètode-kernel.
    El color indica el F1-score aconseguit.
    
    Args:
        directori: Ruta al directori que conté els JSON d'estadístiques
        output_path: Ruta on guardar la imatge (opcional)
    """
    
    # Buscar tots els fitxers JSON d'estadístiques
    fitxers = [f for f in os.listdir(directori) if f.startswith('estadisticas_') and f.endswith('.json')]
    
    if not fitxers:
        print("No s'han trobat fitxers d'estadístiques.")
        return
    
    # Diccionaris per emmagatzemar resultats
    millor_k = {}      # {kernel: {metode: k}}
    millor_f1 = {}     # {kernel: {metode: f1}}
    
    # Processar cada fitxer (cada kernel)
    for fitxer in fitxers:
        # Extreure nom del kernel del nom del fitxer
        kernel = fitxer.replace('estadisticas_', '').replace('.json', '')
        
        filepath = os.path.join(directori, fitxer)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        millor_k[kernel] = {}
        millor_f1[kernel] = {}
        
        # Per cada mètode, trobar la K amb millor F1
        for metode, resultats_k in data.items():
            best_f1 = -1
            best_k = None
            
            for k_str, stats in resultats_k.items():
                f1 = stats['f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_k = k_str
            
            millor_k[kernel][metode] = best_k
            millor_f1[kernel][metode] = best_f1
    
    # Obtenir llistes ordenades de kernels i mètodes
    kernels = sorted(millor_k.keys())
    metodes = sorted(list(millor_k[kernels[0]].keys()))
    
    # Crear matrius per al heatmap
    matriu_k = np.zeros((len(metodes), len(kernels)))
    matriu_f1 = np.zeros((len(metodes), len(kernels)))
    
    for i, metode in enumerate(metodes):
        for j, kernel in enumerate(kernels):
            matriu_k[i, j] = millor_k[kernel].get(metode, 0)
            matriu_f1[i, j] = millor_f1[kernel].get(metode, 0)
    
    # Crear la figura
    plt.figure(figsize=(12, 8))
    
    # Crear heatmap amb seaborn
    # El color es basa en F1, les anotacions mostren K
    ax = sns.heatmap(
        matriu_f1,
        annot=matriu_k,  # Mostrar K com a text
        fmt='.0f',       # Format sense decimals per a K
        cmap='Reds',   # Vermell (dolent) -> Groc -> Verd (bo)
        vmin=0,
        vmax=1,
        xticklabels=kernels,
        yticklabels=metodes,
        cbar_kws={'label': 'F1-Score'},
        annot_kws={'size': 10, 'weight': 'bold'}
    )
    
    # Títol i etiquetes
    plt.title("Millor K per a cada combinació Mètode-Kernel\n(Color = F1-Score, Valor = K òptima)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Kernel SVM", fontsize=12)
    plt.ylabel("Mètode d'extracció", fontsize=12)
    
    # Rotar etiquetes de l'eix X per millor llegibilitat
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    if output_path is None:
        output_path = os.path.join(directori, 'heatmap_millors_k.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap guardat a: {output_path}")
    
    return output_path

def visualizar_metricas(json_path, metodo, K):

    kernel = str((json_path.split('_')[-1].split('.')[0])).upper()
    with open(json_path, 'r') as f:
        data = json.load(f)
    

    if metodo not in data:
        print(f"❌ Método '{metodo}' no encontrado.")
        print(f"Métodos disponibles: {list(data.keys())}")
        return

    K_str = str(K)
    if K_str not in data[metodo]:
        print(f"❌ K={K} no encontrado para el método '{metodo}'.")
        print(f"Valores de K disponibles: {sorted([int(k) for k in data[metodo].keys()])}")
        return

    stats = data[metodo][K_str]

    clases = [c for c in stats['metricas_por_clase'].keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Anàlisi per {metodo} amb K={K} i kernel {kernel}\nAccuracy: {stats["accuracy"]:.4f}, Precision: {stats["precision"]:.4f} \
                 Recall: {stats["recall"]:.4f}, F1-score: {stats["f1"]:.4f}', 
                 fontsize=14, fontweight='bold')
    
    # --- Gráfico 1: Métricas por clase ---
    precision_per_class = [stats['metricas_por_clase'][cls]['precision'] for cls in clases]
    recall_per_class = [stats['metricas_por_clase'][cls]['recall'] for cls in clases]
    f1_per_class = [stats['metricas_por_clase'][cls]['f1-score'] for cls in clases]
    
    x = np.arange(len(clases))
    width = 0.25
    
    axes[0].bar(x - width, precision_per_class, width, label='Precisió', alpha=0.8, color='#1f77b4')
    axes[0].bar(x, recall_per_class, width, label='Recall', alpha=0.8, color='#ff7f0e')
    axes[0].bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8, color='#2ca02c')
    axes[0].set_xlabel('Classe', fontsize=11)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title('Mètriques per Classe', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(clases, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])
    
    # --- Gráfico 2: Matriz de confusión ---
    conf_matrix = np.array(stats['confusion_matrix'])
    im = axes[1].imshow(conf_matrix, cmap='Reds', aspect='auto')
    axes[1].set_xticks(np.arange(len(clases)))
    axes[1].set_yticks(np.arange(len(clases)))
    axes[1].set_xticklabels(clases, rotation=45, ha='right')
    axes[1].set_yticklabels(clases)
    axes[1].set_xlabel('Predicció', fontsize=11)
    axes[1].set_ylabel('Real', fontsize=11)
    axes[1].set_title('Matriu de Confusió', fontsize=12, fontweight='bold')
    
    # Añadir valores en la matriz
    for i in range(len(clases)):
        for j in range(len(clases)):
            text = axes[1].text(j, i, conf_matrix[i, j],
                               ha="center", va="center", 
                               color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black",
                               fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    
    # Guardar
    output_filename = f'metriques_{metodo.replace(",", "_")}_K{K}.png'
    output_path = os.path.join(os.path.dirname(json_path), output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Gràfic guardat a: {output_filename}")
    
    return output_path

def analitzar_roc(X_train, y_train, X_test, y_test, kernel='rbf', K=None, method=None, c = 1):

    clf = svm.SVC(kernel=kernel, C=c)
    clf.fit(X_train, y_train)

    clases = sorted(list(set(y_train)))
    y_scores = clf.decision_function(X_test)
    print(y_scores)
    y_test_bin = label_binarize(y_test, classes=clases)

    plt.figure(figsize=(10, 8))
    colores = [
        '#e41a1c',  # Rojo
        '#377eb8',  # Azul
        '#4daf4a',  # Verde
        '#984ea3',  # Púrpura
        '#ff7f00',  # Naranja
        '#17becf',  # Cyan
        '#a65628',  # Marrón
        '#f781bf',  # Rosa
        '#999999',  # Gris
        '#66c2a5',  # Turquesa
        '#fc8d62',  # Coral
        '#8da0cb',  # Azul lavanda
        '#e78ac3',  # Magenta
        '#a6d854',  # Lima
        '#ffd92f',  # Dorado
        '#e5c494',  # Beige
        '#b3b3b3',  # Gris claro
        '#1b9e77',  # Verde azulado
        '#d95f02',  # Naranja oscuro
        '#7570b3',  # Índigo
        '#e31a1c',  # Rojo carmín
        '#33a02c',  # Verde esmeralda
        '#6a3d9a',  # Violeta oscuro
        '#cab2d6',  # Lavanda
        '#b15928',  # Terracota
    ]

    estils = ['-', '-', '-', '--', '--', ':', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.', '-']
    marcadors = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '+', 'o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '+', 'o', 's', '^', 'D', 'v']
    
    auc_valores = {}
    tpr_list = []
    fpr_comun = np.linspace(0, 1, 100)
    
    for i, clase in enumerate(clases):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        auc_valor = auc(fpr, tpr)
        auc_valores[clase] = auc_valor

        tpr_list.append(np.interp(fpr_comun, fpr, tpr))

        plt.plot(fpr, tpr, 
                        color=colores[i % len(colores)],
                        linestyle=estils[i % len(estils)],
                        marker=marcadors[i % len(marcadors)],
                        markersize=4,
                        markevery=10, 
                        linewidth=2,
                        label=f'{clase} (AUC = {auc_valor:.3f})')

    tpr_media = np.mean(tpr_list, axis=0)
    auc_media = auc(fpr_comun, tpr_media)
    auc_valores['media'] = auc_media


    
    plt.plot(fpr_comun, tpr_media, color='navy', linewidth=3, linestyle='--',
             label=f'Mitjana (AUC = {auc_media:.3f})')
    

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('Tasa de Falsos Positius (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaders Positius (TPR)', fontsize=12)
    
    titulo = f'Curves ROC - Kernel: {kernel}'
    if K:
        titulo += f' - K={K}'
    if method:
        titulo += f' - {method}'
    plt.title(titulo, fontsize=14, fontweight='bold')
    
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    nombre = f'roc_{kernel}'
    if K:
        nombre += f'_K{K}'
    if method:
        nombre += f'_{method.replace(",", "_")}'
    nombre += '.png'
    
    output_path = os.path.join(os.path.dirname(__file__), nombre)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
        
    return auc_valores

def visualitzar_tsne(X,y):
    X,y = filtrar_por_centroide(X,y)
    X = np.array(X)
    y = np.array(y)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X)
    
    # Convertir etiquetas a números
    etiquetas_unicas, y_numerico = np.unique(y, return_inverse=True)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_numerico, 
                          cmap='tab20', alpha=0.6, s=30)
    
    plt.xlabel('Eix X')
    plt.ylabel('Eix Y')
    plt.title('Visualització t-SNE')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()

def filtrar_por_centroide(X, y, porcentaje=0.25):
    X = np.array(X)
    y = np.array(y)
    
    etiquetas_unicas = np.unique(y)  # ['pizza', 'pasta', 'sushi', ...]
    
    X_filtrado = []
    y_filtrado = []
    
    for etiqueta in etiquetas_unicas:  # Para cada clase...
        
        # 1. Obtener solo los puntos de esta clase
        indices = np.where(y == etiqueta)[0]  # Posiciones donde y == 'pizza'
        X_clase = X[indices]  # Descriptores solo de 'pizza'
        
        # 2. Encontrar el centroide con KMeans (K=1 = un solo centro)
        kmeans = MiniBatchKMeans(n_clusters=1, random_state=42)
        kmeans.fit(X_clase)
        centroide = kmeans.cluster_centers_[0]  # El punto "medio" de la clase
        
        # 3. Calcular distancia de cada punto al centroide
        distancias = cdist(X_clase, [centroide], metric='euclidean').flatten()
        # distancias = [0.5, 2.3, 0.8, 5.1, ...]  (una por punto)
        
        # 4. Quedarse con el 25% más cercano
        n_seleccionar = max(1, int(len(X_clase) * porcentaje))  # Si hay 100, selecciona 25
        indices_cercanos = np.argsort(distancias)[:n_seleccionar]  # Índices ordenados por distancia
        
        # 5. Guardar los puntos seleccionados
        X_filtrado.append(X_clase[indices_cercanos])
        y_filtrado.append(np.array([etiqueta] * n_seleccionar))
    
    # 6. Juntar todo en arrays finales
    return np.vstack(X_filtrado), np.concatenate(y_filtrado)

def generar_dendrograma_desde_json(json_path, method, k):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    # Leer JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extraer nombres de clases desde metricas_por_clase
    metricas = data[method][str(k)]['metricas_por_clase']
    clases = [c for c in metricas.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
    clases = sorted(clases)
    
    # Extraer matriz de confusión
    cm = np.array(data[method][str(k)]['confusion_matrix'])
    
    print(f"Clases: {len(clases)}")
    print(f"Matriz: {cm.shape}")
    
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
    plt.figure(figsize=(14, 6))
    dendrogram(Z, labels=clases, leaf_rotation=45)
    plt.xlabel('Clases')
    plt.ylabel('Distancia')
    plt.title(f'Dendrograma de similitud entre clases\n({method}, K={k})')
    plt.tight_layout()
    plt.savefig('dendrograma.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return Z, clases

if __name__ == '__main__':
    #visualitzar_tsne()
    #visualitzar_tots_els_kernels(os.path.dirname(os.path.abspath(__file__)), 'full')
    #generar_heatmap_millors_k(os.path.dirname(os.path.abspath(__file__)))
    visualizar_metricas(os.path.join(os.path.dirname(__file__), 'estadisticas_sigmoide.json'), 'sift,splitted', '2000')
    #generar_dendrograma_desde_json(os.path.join(os.path.dirname(__file__), 'estadisticas_sigmoide.json'),"sift,splitted","2000")