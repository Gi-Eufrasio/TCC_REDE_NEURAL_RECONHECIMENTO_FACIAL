import random
import sys
sys.path.append('CNN/')
from tabnanny import verbose
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import save_model

Classes_e_indices = ''
Atualiza_Pesos_Entrada = 0
Atualiza_Pesos_Saida = 0

def Criacao_Datasets_Atualizacao():

    gerador_treinamento = ImageDataGenerator(rescale=1./255,
                                            rotation_range= 7,
                                            horizontal_flip= True,
                                            zoom_range=0.2)

    dataset_treinamento = gerador_treinamento.flow_from_directory('CNN/Rede_Treinada_MobileNet/Datasets_FEI/Treino_FEI/',
                                                                target_size=(224, 224),
                                                                batch_size= 32,
                                                                class_mode='categorical',
                                                                shuffle=True)

    Classes_e_indices = dataset_treinamento.class_indices

    gerador_test = ImageDataGenerator(rescale=1./255)

    dataset_test = gerador_test.flow_from_directory('CNN/Rede_Treinada_MobileNet/Datasets_FEI/Teste_FEI/',
                                                target_size=(224, 224),
                                                batch_size=1,
                                                class_mode='categorical',
                                                shuffle= False)

    return dataset_treinamento, dataset_test

def carrega_CNN_Modelo():

    Carrega_Rede = tf.keras.models.load_model('CNN/Rede_Treinada_MobileNet/Pesos_Pre_Treino_MobileNet_100_h5_v.h5')

    return Carrega_Rede

def Reconhecimento_BD_Teste_CNN_Treinada_1():
    network_atual_reconhecimento = carrega_CNN_Modelo()
    _, dataset_teste = Criacao_Datasets_Atualizacao()

    Nomenclatura = 'pessoa_' + str(random.randint(0, 99))
    Nomenclatura_id = int(Nomenclatura.split('_')[1].replace(Nomenclatura[0], '').split('.')[0].replace(Nomenclatura[1],''))
    imagem = cv2.imread('CNN/Rede_Treinada_MobileNet/Datasets_FEI/Teste_FEI/' + Nomenclatura + '/'+ str(Nomenclatura_id) +'a.jpg' )
    imagemReconhecimento = imagem

    # Redimensionando a imagem
    imagem = cv2.resize(imagem, (224, 224))

    #normalizando imagem
    imagem = imagem / 255

    imagem = imagem.reshape(-1, 224, 224, 3)

    Reconhecimento = network_atual_reconhecimento.predict(imagem)
    Reconhecimento = np.argmax(Reconhecimento, axis=1)

    Reconhecimento_porcentagem = network_atual_reconhecimento.predict(imagem)
    Reconhecimento_porcentagem = np.max(Reconhecimento_porcentagem)
    #Reconhecimento_porcentagem = (Reconhecimento_porcentagem[Reconhecimento] * 100)
    Reconhecimento_porcentagem = (Reconhecimento_porcentagem * 100)
    Reconhecimento_porcentagem = round(Reconhecimento_porcentagem)

    pessoas_lista_id = []
    Faces_Reconhecidas = []
    indice_reconhecimento = ''

    for X in dataset_teste.class_indices:
            pessoas_lista_id.append(X)

    for Total_Reconhecido in Reconhecimento:
            Faces_Reconhecidas = pessoas_lista_id[Total_Reconhecido]
            indice_reconhecimento = Total_Reconhecido
    
    print('Face Reconhecida, como: ' + Faces_Reconhecidas)
    print('Indice de Reconhecimento: ' + str(indice_reconhecimento))
    print('Probabilidade de: ' + str(Reconhecimento_porcentagem) + '%')

    cv2.putText(imagemReconhecimento, "Classe: " + str(Nomenclatura) ,(10,30),cv2.FONT_HERSHEY_DUPLEX,0.5,255)
    cv2.putText(imagemReconhecimento, "Reconhecimento: " + str(Faces_Reconhecidas), (10,70),cv2.FONT_HERSHEY_DUPLEX,0.5,255)
    cv2.putText(imagemReconhecimento, "Chance: " + str(Reconhecimento_porcentagem) + '%' ,(10,110),cv2.FONT_HERSHEY_DUPLEX,0.5,255)
    cv2.imshow('', imagemReconhecimento)
    cv2.waitKey(0) 
