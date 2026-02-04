from a_local_feature_extraction import creation_of_descriptors, resize, group_classes
from b_vocabulary_extraction import split_train_test, build_vocabulary
from c_convert_nontabular_to_tabular import dataset_to_histograms, \
    calcular_estadisticas, guardar_estadisticas_json
from d_metric_visualization import visualitzar_tots_els_kernels, generar_heatmap_millors_k, analitzar_roc, visualitzar_tsne, filtrar_por_centroide
from sklearn import svm
import os
import json
import numpy as np



def SVCRbf(method, K, X_train, y_train, X_test, y_test, c_flag, dim_flag, c, filename = 'estadisticas_rbf.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if c_flag == True:
                    if method in data and str(c) in data[method]:
                        print(f"c={c} con método {method} ya procesada, saltant...")
                        return
                elif dim_flag == True:
                    if method in data and str(K) in data[method]:
                        print(f"dim={K} con método {method} ya procesada, saltant...")
                        return
                elif c_flag == False:
                    if method in data and str(K) in data[method]:
                        print(f"K={K} con método {method} ya procesada, saltant...")
                        return
                
        
        clf = svm.SVC(kernel='rbf', C=c)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if c_flag == True:
            guardar_estadisticas_json(method, c, calcular_estadisticas(y_test, y_pred), filename = filename)
        elif dim_flag == True:
             guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)
             
        elif c_flag == False:
            guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred), filename = filename)


def SVCSigmoid(method, K, X_train, y_train, X_test, y_test, c_flag, dim_flag, c, filename = 'estadisticas_sigmoide.json'):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if c_flag == True:
                    if method in data and str(c) in data[method]:
                        print(f"c={c} con método {method} ya procesada, saltant...")
                        return
                elif dim_flag == True:
                    if method in data and str(K) in data[method]:
                        print(f"dim={K} con método {method} ya procesada, saltant...")
                        return
                elif c_flag == False:
                    if method in data and str(K) in data[method]:
                        print(f"K={K} con método {method} ya procesada, saltant...")
                        return
        clf = svm.SVC(kernel='sigmoid', C=c, class_weight = 'balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)


        if c_flag == True:
            guardar_estadisticas_json(method, c, calcular_estadisticas(y_test, y_pred, y_train, y_pred_train), filename = filename)
        elif dim_flag == True:
            guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred, y_train, y_pred_train), filename = filename)
        elif c_flag == False:
            guardar_estadisticas_json(method, K, calcular_estadisticas(y_test, y_pred, y_train, y_pred_train), filename = filename)

def calcular_metricas(porcentajes, group, dim_descriptors, max_keypoints, num_clases, resized, images_for_class, methods,
                      k, c_flag = False, descriptors_flag = False, split_flag = False, vocab_flag = False, kernel = 'rbf', dim_flag = False):
    for method in methods:
        creation_of_descriptors(resized, [method], flag = descriptors_flag,
                                dim_descriptors = dim_descriptors,
                                max_keypoints = max_keypoints, num_classes = num_clases, images_for_class= images_for_class)

        train_data, test_data = split_train_test(split_flag, method, dim_descriptors, max_keypoints)
        train_data, test_data = group_classes(group, train_data, test_data)
        

        for K in k:
            if c_flag == True:
                c = np.arange(0.1,5, 0.1).astype('float')
                vocabulary  = build_vocabulary(train_data, method, vocab_flag, K)
                for C in c:
                    C = round(C, 2)
                    X_train, y_train = dataset_to_histograms(train_data, vocabulary, K)
                    X_test, y_test = dataset_to_histograms(test_data, vocabulary, K)
                    if kernel == 'rbf':
                        SVCRbf(method, K, X_train, y_train, X_test, y_test,c_flag, C)
                    elif kernel == 'sigmoid':
                        SVCSigmoid(method, K, X_train, y_train, X_test, y_test,c_flag, False, C)
                return
            
            elif dim_flag == True:
                    creation_of_descriptors(True, [method], flag = True,
                                dim_descriptors = K,
                                max_keypoints = max_keypoints, num_classes = num_classes, images_for_class= images_for_class)
                    train_data, test_data = split_train_test(True, method, K, max_keypoints)
                    vocabulary1  = build_vocabulary(train_data, method, True, 1900)
                    X_train1, y_train1 = dataset_to_histograms(train_data, vocabulary1, 1900)
                    X_test1, y_test1 = dataset_to_histograms(test_data, vocabulary1, 1900)
                    SVCRbf(method, K, X_train1, y_train1, X_test1, y_test1,c_flag, True, 1.9)
                    vocabulary2  = build_vocabulary(train_data, method, True, 2000)
                    X_train2, y_train2 = dataset_to_histograms(train_data, vocabulary2, 2000)
                    X_test2, y_test2 = dataset_to_histograms(test_data, vocabulary2, 2000)
                    SVCSigmoid(method, K, X_train2, y_train2, X_test2, y_test2,c_flag, True, 1.5)
                    continue
                 
                     
            C = 1.5 
            vocabulary  = build_vocabulary(train_data, method, vocab_flag, 2000)
            X_train, y_train = dataset_to_histograms(train_data, vocabulary, 2000)
            X_train, y_train = filtrar_por_centroide(X_train, y_train, porcentaje=porcentaje)
            X_test, y_test = dataset_to_histograms(test_data, vocabulary, 2000)
            if kernel == 'rbf':
                        SVCRbf(method, K, X_train, y_train, X_test, y_test,c_flag, C)
            elif kernel == 'sigmoid':
                        SVCSigmoid(method, K , X_train, y_train, X_test, y_test, c_flag, False,C)

def calcular_roc_curves(flag_resized, method, flag_descriptors, dim_descriptors, max_keypoints, num_classes, images_for_class,
                        flag_split, flag_kmeans, k, kernel, c = 1):
    creation_of_descriptors(flag_resized, [method], flag = flag_descriptors,
                                dim_descriptors = dim_descriptors,
                                max_keypoints = max_keypoints, num_classes = num_classes, images_for_class= images_for_class)

    train_data, test_data = split_train_test(flag_split, method, dim_descriptors, max_keypoints)
    vocabulary  = build_vocabulary(train_data, method, flag_kmeans, k)
    X_train, y_train = dataset_to_histograms(train_data, vocabulary, k)
    X_test, y_test = dataset_to_histograms(test_data, vocabulary, k)
    analitzar_roc(X_train, y_train, X_test, y_test, kernel=kernel, K = k, method=method, c = c)

if __name__ == '__main__':
    
    lista_capetas = os.listdir(os.path.join(os.path.dirname(__file__), ".."))
    if 'Food Classification resized' not in lista_capetas:
        resize((300,300))
    num_classes = None
    methods = ['sift,splitted']
    group = [['chai','kulfi'], ['butter_naan','chapati']]
    #for method in methods:
        #creation_of_descriptors(True,[method],True,0,0,None,0)
        #split_train_test(True, method, 0,0)
    k = [2000]
    porcentaje = 1
    calcular_metricas(porcentaje, group,0,0,num_classes, True, 0, methods, k, c_flag = False, descriptors_flag = False, split_flag = False, vocab_flag = True, kernel='sigmoid', dim_flag = False)

    #calcular_roc_curves(True, 'sift,splitted', False, 0, 0, num_classes, 0, False, False, 2000, 'sigmoid', 1.5)

    
