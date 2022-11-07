import sys
sys.path.append('CNN/')
sys.path.append('sqlite_RC/')
from Datasets_Bases_CNN import carrega_CNN_Modelo, Criacao_Datasets_Atualizacao
from BD_Insert import Consulta_BD_Total_Usuario_Especifico
import numpy as np
import cv2

def Reconhecimento_BD_Unico_Imagem(Imagem_Reconhecimento):

    imagem = cv2.imread(Imagem_Reconhecimento)
    print(imagem)

    # Redimensionando a imagem
    imagem = cv2.resize(imagem, (224, 224))

    #normalizando imagem
    imagem = imagem / 255

    imagem = imagem.reshape(-1, 224, 224, 3)

    #Reconhecimento Facial
    network_atual_reconhecimento = carrega_CNN_Modelo()
    _, dataset_teste = Criacao_Datasets_Atualizacao()

    Reconhecimento = network_atual_reconhecimento.predict(imagem)
    Reconhecimento = np.argmax(Reconhecimento, axis=1)

    Reconhecimento_porcentagem = network_atual_reconhecimento.predict(imagem)
    Reconhecimento_porcentagem = np.max(Reconhecimento_porcentagem)
    Reconhecimento_porcentagem = (Reconhecimento_porcentagem * 100)
    Reconhecimento_porcentagem = round(Reconhecimento_porcentagem, 1)

    if Reconhecimento_porcentagem >= 90:
        print('Reconheceu')

        pessoas_lista_id = []
        Faces_Reconhecidas = []
        indice_reconhecimento = ''

        for X in dataset_teste.class_indices:
            pessoas_lista_id.append(X)

        for Total_Reconhecido in Reconhecimento:
            Faces_Reconhecidas = pessoas_lista_id[Total_Reconhecido]
            indice_reconhecimento = Total_Reconhecido
            
        Consulta = Consulta_BD_Total_Usuario_Especifico("'" + Faces_Reconhecidas + "'")
        print('Face Reconhecida, como: ' + Faces_Reconhecidas)
        print('Indice de Reconhecimento: ' + str(indice_reconhecimento))
        print('Usuario Reconhecido, como: ' + Consulta)
        print('Probabilidade de: ' + str(Reconhecimento_porcentagem) + '%')

    else:
        Consulta = 'Não Identificado!'
        print('Probabilidade de: ' + str(Reconhecimento_porcentagem) + '%')
        print('Face não identificada')
    
    return Consulta, Faces_Reconhecidas

def Reconhecimento_BD_Unico_Video(Imagem_Reconhecimento):

    imagem = Imagem_Reconhecimento
    print(imagem)

    # Redimensionando a imagem
    imagem = cv2.resize(imagem, (224, 224))

    #normalizando imagem
    imagem = imagem / 255

    imagem = imagem.reshape(-1, 224, 224, 3)

    #Reconhecimento Facial
    network_atual_reconhecimento = carrega_CNN_Modelo()
    _, dataset_teste = Criacao_Datasets_Atualizacao()

    Reconhecimento = network_atual_reconhecimento.predict(imagem)
    Reconhecimento = np.argmax(Reconhecimento, axis=1)

    Reconhecimento_porcentagem = network_atual_reconhecimento.predict(imagem)
    Reconhecimento_porcentagem = np.max(Reconhecimento_porcentagem)
    Reconhecimento_porcentagem = (Reconhecimento_porcentagem * 100)
    Reconhecimento_porcentagem = round(Reconhecimento_porcentagem, 1)

    if Reconhecimento_porcentagem >= 85:
        print('Reconheceu')

        pessoas_lista_id = []
        Faces_Reconhecidas = []
        indice_reconhecimento = ''

        for X in dataset_teste.class_indices:
            pessoas_lista_id.append(X)

        for Total_Reconhecido in Reconhecimento:
            Faces_Reconhecidas = pessoas_lista_id[Total_Reconhecido]
            indice_reconhecimento = Total_Reconhecido
            
        Consulta = Consulta_BD_Total_Usuario_Especifico("'" + Faces_Reconhecidas + "'")
        print('Face Reconhecida, como: ' + Faces_Reconhecidas)
        print('Indice de Reconhecimento: ' + str(indice_reconhecimento))
        print('Usuario Reconhecido, como: ' + Consulta)
        print('Probabilidade de: ' + str(Reconhecimento_porcentagem) + '%')

    else:
        
        Consulta = 'Não Identificado!'
        print('Probabilidade de: ' + str(Reconhecimento_porcentagem) + '%')
        print('Face não identificada')
    
    return Consulta

