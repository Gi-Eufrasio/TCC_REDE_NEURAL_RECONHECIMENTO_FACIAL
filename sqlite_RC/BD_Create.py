import sqlite3

conn = sqlite3.connect('Usuarios_Faciais.db')

cursor = conn.cursor()

cursor.execute(""" CREATE TABLE USUARIOS(
                                     ID INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                                     ID_USUARIO_SISTEMA TEXT NOT NULL,
                                     USUARIO_ID TEXT NOT NULL
);
""")

print('Tabela Criada')

conn.close
