import sqlite3

def insercao_BD(ID_Usuario_Pre_Definido, Usuario_Definido):

    conn = sqlite3.connect('Usuarios_Faciais.db')

    cursor = conn.cursor()

    cursor.execute(" INSERT INTO USUARIOS (ID_USUARIO_SISTEMA, USUARIO_ID) VALUES ("+ "'" + ID_Usuario_Pre_Definido + "'," + "'"+ Usuario_Definido +"'" +")")

    conn.commit()

    print('Inserção Realizada')

    conn.close

def Consulta_BD_Total():

    conn = sqlite3.connect('Usuarios_Faciais.db')

    cursor = conn.cursor()

    cursor.execute(""" SELECT * FROM USUARIOS;

    """)

    for linha in cursor.fetchall():
        print(linha)

    conn.close


def Consulta_BD_Total_Usuario_Especifico(ID_Usuario_Pre_Definido):

    conn = sqlite3.connect('Usuarios_Faciais.db')

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM USUARIOS US WHERE ID_USUARIO_SISTEMA = "+ ID_Usuario_Pre_Definido +";")

    for linha in cursor.fetchall():

        Consulta = linha[2]

    conn.close
    return Consulta

def Exclui_e_Cria_BD_Novamente():
    conn = sqlite3.connect('Usuarios_Faciais.db')

    cursor = conn.cursor()

    cursor.execute("DROP TABLE USUARIOS")
    print('Dropo')

    #Após Dropar Recria Tabela na Base.
    cursor.execute(""" CREATE TABLE USUARIOS(
                                     ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                                     ID_USUARIO_SISTEMA TEXT NOT NULL,
                                     USUARIO_ID TEXT NOT NULL
    );
    """)

    print('Crio novamente')

    conn.close