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

    dataset_treinamento = gerador_treinamento.flow_from_directory('CNN/Faces_Capturadas/Treino/',
                                                                target_size=(224, 224),
                                                                batch_size= 16,
                                                                class_mode='categorical',
                                                                shuffle=True)

    Classes_e_indices = dataset_treinamento.class_indices

    gerador_test = ImageDataGenerator(rescale=1./255)

    dataset_test = gerador_test.flow_from_directory('CNN/Faces_Capturadas/Teste/',
                                                target_size=(224, 224),
                                                batch_size=1,
                                                class_mode='categorical',
                                                shuffle= False)

    return dataset_treinamento, dataset_test

def CNN_treino_e_Atualizacao():

    #Chamada dos datasets de treinamento e teste
    dataset_treinamento, dataset_test = Criacao_Datasets_Atualizacao()

    # Caso não exista o modelo, ee continua para poder colocar um novo modelo no lugar
    try :
        #Chamando Modelo de CNN treinado
        network_loaded =  carrega_CNN_Modelo()
    except:
        pass

    #Definição do modelo base da rede neural pré-treinada
    modelo_base = tf.keras.applications.MobileNet(weights='imagenet', include_top = False,
                                            input_tensor= Input(shape=(224, 224, 3)))

    #Fine_Tuning para melhorar o desempenho da rede neural
    fine_tuning_at = 76
    modelo_base.trainable=True
    for layer in modelo_base.layers[:fine_tuning_at]:
        layer.trainable = False
        print(layer, layer.trainable)


    #Atualização neuronios e classes 
    Atualiza_neuronios_Saida = len(dataset_treinamento.class_indices)

    #Atualização da MLP Dense
    head_model = modelo_base.output
    head_model = GlobalAveragePooling2D()(head_model)
    head_model = Dense(units= 1050, activation='relu')(head_model)
    head_model = Dropout(rate=0.2)(head_model)
    head_model = Dense(units= 1050, activation='relu')(head_model)
    head_model = Dropout(rate=0.2)(head_model)
    head_model = Dense(units= int(Atualiza_neuronios_Saida), activation= 'softmax')(head_model)

    network_loaded = Model(inputs = modelo_base.input, outputs = head_model)

    network_loaded.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treino
    historico = network_loaded.fit(dataset_treinamento, epochs=50, verbose=1)
    print(historico)

    salvando_novo_modelo_CNN(network_loaded)

def CNN_Treina_Multiplas_Vezes():

    dataset_treinamento, _ = Criacao_Datasets_Atualizacao()

    Carrega_Rede = tf.keras.models.load_model('CNN/Algoritmo_Pesos_Pre_Treinados_MobileNet/Pesos_Pre_Treino_MobileNet_100_h5_v.h5')

    historico =  Carrega_Rede.fit(dataset_treinamento, epochs=15)

    Carrega_Rede.save('CNN/Algoritmo_Pesos_Pre_Treinados_MobileNet/Pesos_Pre_Treino_MobileNet_100_h5_v.h5')

def carrega_CNN_Modelo():

    #Carregando json da CNNs
    with open('CNN/Algoritmo_Pesos_Pre_Treinados_MobileNet/network_Pre_Treino_MobileNet_100_v.json') as json_file:
        json_saved_model = json_file.read()
    
    # Carregando as bias da CNN
    network_loaded = tf.keras.models.model_from_json(json_saved_model)
    network_loaded.load_weights('CNN/Algoritmo_Pesos_Pre_Treinados_MobileNet/Pesos_Pre_Treino_MobileNet_100_hdf5_v.hdf5')
    network_loaded.compile(loss= 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

    return network_loaded

def salvando_novo_modelo_CNN(network_loaded):

    # Salvando novo modelo de CNN (Sobrepondo)
    # Salvando Json
    model_json = network_loaded.to_json()
    with open('CNN/Algoritmo_Pesos_Pre_Treinados_MobileNet/network_Pre_Treino_MobileNet_100_v.json', 'w') as json_file:
        json_file.write(model_json)

    # Salvando hdf5 Pesos
    network_saved = save_model(network_loaded, 'CNN/Algoritmo_Pesos_Pre_Treinados_MobileNet/Pesos_Pre_Treino_MobileNet_100_hdf5_v.hdf5')

    # Salvando h5 Pesos
    network_loaded.save('CNN/Algoritmo_Pesos_Pre_Treinados_MobileNet/Pesos_Pre_Treino_MobileNet_100_h5_v.h5')
