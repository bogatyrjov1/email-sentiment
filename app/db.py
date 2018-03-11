import mysql.connector
from conf import conf_sandbox

config = {
    'user': conf_sandbox["db"]["user"],
    'password': conf_sandbox["db"]["password"],
    'host': conf_sandbox["db"]["host"],
    'database': conf_sandbox["db"]["database"],
    'raise_on_warnings': True,
}

cnx = mysql.connector.connect(**config)

cnx.close()
