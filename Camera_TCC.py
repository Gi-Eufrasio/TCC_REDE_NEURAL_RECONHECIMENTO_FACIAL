import cv2
import sys
import os
import shutil
import keyboard
sys.path.append('CNN/')
sys.path.append('CNN/Rede_Treinada_MobileNet')
sys.path.append('CNN/Rede_Treinada_MobileNet_Varias_Posicoes')
sys.path.append('sqlite_RC/')
from CNN_FEI import Reconhecimento_BD_Teste_CNN_Treinada_1
from CNN_FEI_14_Posicoes import Reconhecimento_BD_Teste_CNN_Treinada_Varias_Posicoes
from Reconhecimento import Reconhecimento_BD_Unico_Imagem, Reconhecimento_BD_Unico_Video
from Datasets_Bases_CNN import CNN_treino_e_Atualizacao, CNN_Treina_Multiplas_Vezes
import time
from BD_Insert import insercao_BD, Exclui_e_Cria_BD_Novamente

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

Dicionario_Usuarios = {}

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

Nome_Reconhecido = ''

Imagens_Capturadas_lista = []

def Novas_Diretorio_Face():
    
    path_treino = 'CNN/Faces_Capturadas/Treino/'
    arquivos = os.listdir(path_treino)
    ids_arquivo = []
    id = 1

    for file in arquivos:
        ids = int(file.split('_')[1].replace(file[0], '').split('.')[0].replace(file[1],''))
        ids_arquivo.append(ids)

    for ids_X in sorted(ids_arquivo):
     
        if ids_X == id:
            id = id + 1
        else:
            id = id

    os.mkdir('CNN/Faces_Capturadas/Treino/pessoa_' + str(id))
    os.mkdir('CNN/Faces_Capturadas/Teste/pessoa_' + str(id))
    print('Diretorio criado id: pessoa_' + str(id))
    Usuario_ID = input('Cadastre o ID Usuario: ')

    dicionario = str(Usuario_ID)
    id_Sistema = 'pessoa_' + str(id)

    insercao_BD(id_Sistema ,dicionario)

def Cadastra_Novas_Faces_no_Usuario(imagem_capturada):

    usuario = input('Insira o id pessoa: ')
    path_treino = 'CNN/Faces_Capturadas/Treino/' + usuario
    ids_arquivo = []
    id = 1

    for imagens_processo_cadastro in imagem_capturada:

        arquivos = os.listdir(path_treino)

        for file in arquivos:
            ids = int(file.split('_')[3].replace(file[0], '').split('.')[0].replace(file[1], ''))
            ids_arquivo.append(ids)

        for ids_X in sorted(ids_arquivo):
        
            if ids_X == id:
                id = id + 1
            else:
                id = id

        cv2.imwrite('CNN/Faces_Capturadas/Treino/' + usuario + '/imagemCapturada_' + usuario + '_' + str(id) +'.png', imagens_processo_cadastro)
        cv2.imwrite('CNN/Faces_Capturadas/Teste/' + usuario + '/imagemCapturada_' + usuario + '_' + str(id) +'.png', imagens_processo_cadastro)

def Processo_Reconhecimento_Salva_Face_Reconhecida(usuario, imagem_capturada):
 
    path_treino = 'CNN/Faces_Capturadas/Treino/' + usuario
    arquivos = os.listdir(path_treino)
    ids_arquivo = []
    id = 1

    for file in arquivos:
        ids = int(file.split('_')[3].replace(file[0], '').split('.')[0].replace(file[1], ''))
        ids_arquivo.append(ids)

    for ids_X in sorted(ids_arquivo):
     
        if ids_X == id:
            id = id + 1
        else:
            id = id
    cv2.imwrite('CNN/Faces_Capturadas/Treino/' + usuario + '/imagemCapturada_' + usuario + '_' + str(id) +'.png', imagem_capturada)
    cv2.imwrite('CNN/Faces_Capturadas/Teste/' + usuario + '/imagemCapturada_' + usuario + '_' + str(id) +'.png', imagem_capturada)

def Reconhecimento_Facial_imagem(imagem_capturada):
    path_treino = 'CNN/Faces_Capturadas/Faces_Capturadas_Processo_Reconhecimento/'
    arquivos = os.listdir(path_treino)
    ids_arquivo = []
    id = 1

    for file in arquivos:
        ids = int(file.split('_')[1].replace(file[0], '').split('.')[0].replace(file[1],''))
        ids_arquivo.append(ids)

    for ids_X in sorted(ids_arquivo):
     
        if ids_X == id:
            id = id + 1
        else:
            id = id

    cv2.imwrite('CNN/Faces_Capturadas/Faces_Capturadas_Processo_Reconhecimento/imagemCapturada_' + str(id) +'.png', imagem_capturada)

    time.sleep(2)

    Reconhecimento_Url = 'CNN/Faces_Capturadas/Faces_Capturadas_Processo_Reconhecimento/imagemCapturada_' + str(id) +'.png'
    Usuario_Reconhecido, Faces_Reconhecidas = Reconhecimento_BD_Unico_Imagem(Reconhecimento_Url) 

    time.sleep(2)
    print(Usuario_Reconhecido)
    Reconhecimento_Questiona_Usuario = input('Esta correto seu Reconhecimento Usuario: T(True) or F(False): ')

    if Reconhecimento_Questiona_Usuario == 'T':
        Processo_Reconhecimento_Salva_Face_Reconhecida(Faces_Reconhecidas, imagem_capturada)
        os.remove(Reconhecimento_Url)
        print('Reconheceu')
    
    elif Reconhecimento_Questiona_Usuario == 'F':
         print('Não Reconheceu!. Digite Usuario correto, ou Cadastre usuario e Face')

    return Faces_Reconhecidas

def Reconhecimento_Facial_Video(imagem_capturada):
    Faces_Reconhecidas = Reconhecimento_BD_Unico_Video(imagem_capturada)
    print(type(imagem_capturada))
    return Faces_Reconhecidas

def Exclui_Diretorios_Recria():

    shutil.rmtree('CNN/Faces_Capturadas/Teste/')
    shutil.rmtree('CNN/Faces_Capturadas/Treino/')
    shutil.rmtree('CNN/Faces_Capturadas/Faces_Capturadas_Processo_Reconhecimento/')
    os.mkdir('CNN/Faces_Capturadas/Teste/')
    os.mkdir('CNN/Faces_Capturadas/Treino/')
    os.mkdir('CNN/Faces_Capturadas/Faces_Capturadas_Processo_Reconhecimento/')

print('----------------------------------------------------------------------------')
print('| W - Adiciona Imagens Faciais                                              |')
print('| A - Cadastra uma nova face (classe)                                       |')
print('| D - Reconhecimento Facial através de imagens                              |')
print('| R - Reconhecimento Facial Video                                           |')
print('| E - Retira o reconhecimento do video                                      |')
print('| T - Treina e atualiza a rede neural                                       |')
print('| Y - Treina multiplas vezes com as mesmas classes                          |')
print('| F - Reconhecimento Facial de imagens do banco, treinamento simples        |')
print('| G - Reconhecimento Facial de imagens do banco, treinamento com 14 posicoes|')
print('| B - Exclui BD e banco de imagens                                          |')
print('| Q - Sai do sistema                                                        |')
print('----------------------------------------------------------------------------')

while True:

    #captura de video

    ret, frame = video_capture.read()
    _, frame_Captura = video_capture.read()

    #conversão de imagem, em tons de cinza para o reconhecimendo haarcascade
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detecção facial utilizando o haarcascade
    detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.5, minSize=(100,100))

    #Retenculo e desenho do nome, do usuario reconhecido
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, Nome_Reconhecido, (x,y +(w+30)), font, 2, (0,255,0))
    
    #Inserção de imagens
    if keyboard.is_pressed('w'):

        print('Registre três imagens do seu rosto caso cadastre um novo usuario!')

        Imagens_Capturadas_lista.append(frame_Captura)
        
        print('Imagem Registrada')

        time.sleep(2)

        if len(Imagens_Capturadas_lista) == 3:
            print('Registrado, Pressione "a" para cadastrar!')
        
        elif len(Imagens_Capturadas_lista) == 2:
            print('Falta registrar uma imagem!')
            print(len(Imagens_Capturadas_lista))

        else: 
            print('Falta registrar duas imagens!')
            print(len(Imagens_Capturadas_lista))

    #Cadastra novas faces no banco de imagens, além de um usuario no banco
    elif keyboard.is_pressed('a'):

        imagem_capturada = Imagens_Capturadas_lista
        Imagens_Capturadas_lista = []

        Novas_Diretorio_Face()
        Cadastra_Novas_Faces_no_Usuario(imagem_capturada)
        print('Diretorio e Imagem Salva')

    #Efetua o reconhecimento facial atrvés de imagens tiradas em video 
    elif keyboard.is_pressed('d'):

        imagem_capturada = frame_Captura
        Reconhecimento_Facial_imagem(imagem_capturada)

    #Efetua reconhecimento facial no video, desenhando o nome do usuario no video
    elif keyboard.is_pressed('r'):
        
        RFV_Reconhecimento = Reconhecimento_Facial_Video(frame_Captura)
        Nome_Reconhecido = RFV_Reconhecimento
    
    elif keyboard.is_pressed('e'):

        Nome_Reconhecido = ''

    #Treina a rede neural, caso o treinamento automatico não temnha sido efetuado
    elif keyboard.is_pressed('t'):
        CNN_treino_e_Atualizacao()

    #Caso esteja apresentando, problemas de reconhecimento
    elif keyboard.is_pressed('y'):

        CNN_Treina_Multiplas_Vezes()

    #Exclui BD e Diretorios do BD de imagens e recria!
    elif keyboard.is_pressed('b'):

        Exclui_e_Cria_BD_Novamente()
        Exclui_Diretorios_Recria()

    #Efetua reconhecimento de imagens do dataset que foi utilizado para o treinamento da RN
    elif keyboard.is_pressed('f'):

        Reconhecimento_BD_Teste_CNN_Treinada_1()

    elif keyboard.is_pressed('g'):

        Reconhecimento_BD_Teste_CNN_Treinada_Varias_Posicoes()

    #Sai do programa
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break    

    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()