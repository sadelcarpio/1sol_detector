from funciones_creadas import *
import cv2 as cv
import os
from statistics import *
import matplotlib.pyplot as plt
import argparse

# Programa: CLASIFICAR SOL/NOSOL
# INPUT -> IMAGEN  (SOL O NO SOL)
# SE COMPARA CON VARIOS TEMPLATES (RANDOM DE "N") -> SE OBTIENE UN SCORE
# SE PROMEDIA
# SI SCORE > THRESHOLD:
# ES SOL
# ELSE:
# NO ES SOL

# Programa: ENTRENAMIENTO/OBTENCIÓN DE THRESHOLD
# PARA: CADA IMAGEN EN SOL
# COMPARAR CON LAS DEMÁS (O AL MENOS "M") -> OBTENER SCORE
# END
# DEL ARRAY DE SCORES, ENCONTRAR SUS MÉTRICAS, HALLAR PROMEDIO
# O MÉTRICAS SIGNIFICATIVAS
# ESTABLECER THRESHOLD

# Definir la entrada por línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Probar imagenes individuales o en grupo")
parser.add_argument("--img", help="Imagen a probar")
args = parser.parse_args()

PREPROCESSING_SIZE = (720, 720)

src_imgs = '../DATA_2/SOL'
dst_imgs = '../DATA_2/SOL_CROP'

filenames = os.listdir(src_imgs)

train_set = []

i = 1
for filename in filenames:
    sol_img = cv.imread(src_imgs + '/' + filename)
    sol_crop = crop_coin_rm_bg(sol_img, PREPROCESSING_SIZE)
    cv.imwrite(dst_imgs + '/' + 'sol_{}.jpg'.format(i), sol_crop)
    train_set.append(sol_crop)
    i += 1

# ENTRENAMIENTO: Hallar el threshold

train_scores = get_scores(train_set, len(train_set)-1)
threshold = mean(train_scores)-stdev(train_scores)
print('Threshold: ' + str(threshold))


# TESTING: Se prueba la imagen desconocida con el threshold escogido

if args.mode == 'individual':
    # Evaluar el modelo en la imagen de prueba
    num_imgs = os.listdir('../DATA_2/PRUEBA')
    test_img_path = '../DATA_2/PRUEBA/' + args.img
    test_img = cv.imread(test_img_path)
    test_img_crop = crop_coin_rm_bg(test_img, PREPROCESSING_SIZE)
    test_score = compare(test_img_crop, train_set)  # random.sample(train_set, n)
    es_1sol = 'Sí' if (test_score > threshold) else 'No'
    plt.axis("off")
    plt.title(es_1sol + ':' + str(round(test_score, 4)) + ', Threshold: ' + str(round(threshold, 4)),
              fontdict={'fontsize': 10})
    plt.imshow(test_img_crop)
    plt.show()

elif args.mode == 'grupal':
    test_imgs_path = '../DATA_2/PRUEBA'
    num_imgs = len(os.listdir(test_imgs_path))
    plt.figure(figsize=(10, 10))
    c = 1
    for i in range(num_imgs):
        path_test = test_imgs_path + '/prueba_{}.jpg'.format(i+1)
        test_img = cv.imread(path_test)
        test_img_crop = crop_coin_rm_bg(test_img, PREPROCESSING_SIZE)
        test_score = compare(test_img_crop, train_set)
        es_1sol = 'Sí' if (test_score > threshold) else 'No'
        plt.subplot(4, 4, c)
        plt.axis('off')
        plt.title(es_1sol + ':' + str(round(test_score, 4)), fontdict={'fontsize': 10})
        plt.imshow(test_img_crop)
        c += 1
    plt.show()
