import mysql.connector
from datetime import datetime


def logUdate(label):
	# establish connection 
	database = mysql.connector.connect( 
	    host="localhost", 
	    user="root", 
	    password="admin123", 
	    database="mainSys"
	) 
	  
	# creating cursor object 
	cur_object = database.cursor() 
	find = "SELECT * FROM data_logs WHERE fruitOrVegetable LIKE %s limit 1" 
	cur_object.execute(find, ("%" + label + "%",)) 
	data = cur_object.fetchall()

	if data : 
		for i in data:
		    count=int(i[2])
		    count = count + 1
		    now = datetime.now()
		    dateString = now.strftime('%Y-%m-%d %H:%M:%S')
		    sql = """ UPDATE data_logs SET counts = %s, lastUpdate = %s WHERE idlogs = %s """
		    data = (count, dateString, i[0])
		    cur_object.execute(sql, data)
		    affected_rows = cur_object.rowcount
		    database.commit()
		    print("Update Sucessful")
	else :
	    print("Not Found")
	    now = datetime.now()
	    dateString = now.strftime('%Y-%m-%d %H:%M:%S')
	    sql = "INSERT INTO data_logs (fruitOrVegetable, counts, lastUpdate) VALUES (%s, %s, %s)"
	    val = (label, "1", dateString)
	    cur_object.execute(sql, val)
	    database.commit()
	    print("New data added")


logUdate("Strawberry")	    