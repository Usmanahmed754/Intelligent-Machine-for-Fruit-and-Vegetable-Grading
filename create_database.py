import mysql.connector

connection = mysql.connector.connect(
  host="localhost",
  user="root",
  password="admin123"
)

cur = connection.cursor()
cur.execute("CREATE DATABASE mainSys")
connection.commit()
connection.close()

connection = mysql.connector.connect(
  host="localhost",
  user="root",
  password="admin123",
  database="mainSys"
)
cur = connection.cursor()

#Dropping users table if already exists.
cur.execute("DROP TABLE IF EXISTS mainSys")

#Creating table as per requirement
sql ='''CREATE TABLE users(
   Id INT PRIMARY KEY AUTO_INCREMENT NOT NULL,
   FirstName CHAR(100) NOT NULL,
   LastName CHAR(100) NOT NULL,
   EMAIL CHAR(100) NOT NULL,
   Password CHAR(100) NOT NULL
)'''

cur.execute(sql)
connection.close()
