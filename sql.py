import datetime
import sqlite3

class SQL:
    def __init__(self):
        self.connection = sqlite3.connect("expenses.db")
        self.cursour = self.connection.cursor()

    def createTables(self):
        self.cursour.execute("CREATE TABLE IF NOT EXISTS expenses(id, name, category, value, year, month, day)")
        self.cursour.execute("CREATE TABLE IF NOT EXISTS conversations(id, question, response, date)")
        self.connection.commit()

    def dropTables(self):
        self.cursour.execute("DROP TABLE expenses")
        self.cursour.execute("DROP TABLE conversations")
        self.connection.commit()

    def getExpenses(self):
        raw_result = self.cursour.execute("SELECT * FROM expenses")
        parsed_result = [{
            'id': x[0],
            'name': x[1],
            'category': x[2],
            'value': x[3],
            'year': x[4],
            'month': x[5],
            'day': x[6],
        } for x in raw_result.fetchall()]
        return parsed_result
    
    def getConversations(self):
        raw_result = self.cursour.execute("SELECT * FROM conversations")
        parsed_result = [{
            'id': x[0],
            'question': x[1],
            'response': x[2],
            'date': x[3],
        } for x in raw_result.fetchall()]
        return parsed_result

    def insertExpense(self, name, category, value, year, month, day):
        time = datetime.datetime.now().isoformat()
        data = (time, name, category, value, year, month, day)
        self.cursour.execute("INSERT INTO expenses VALUES(?, ?, ?, ?, ?, ?, ?)", data)
        self.connection.commit()
        return  {
            'id': data[0],
            'name': data[1],
            'category': data[2],
            'value': data[3],
            'year': data[4],
            'month': data[5],
            'day': data[6],
        }

    def insertConversation(self, question, response):
        time = datetime.datetime.now().isoformat()
        data = (time, question, response, time)
        self.cursour.execute("INSERT INTO conversations VALUES(?, ?, ?, ?)", data)
        self.connection.commit()
        return {
            'id': data[0],
            'question': data[1],
            'response': data[2],
            'date': data[3],
        }

    def deleteExpense(self, id):
        self.cursour.execute("DELETE FROM expenses WHERE ID = ?", (id,))
        self.connection.commit()    
        
    def deleteConversation(self, id):
        self.cursour.execute("DELETE FROM conversations WHERE ID = ?", (id,))
        self.connection.commit()

    def executeQuery(self, query):
        raw_result = self.cursour.execute(query).fetchall()
        self.connection.commit()

        if (len(raw_result) == 1 and len(raw_result[0]) == 1):
            return raw_result[0][0]
        
        return raw_result

sql = SQL()
# sql.dropTables()
# sql.createTables()

# res = sql.executeQuery('DELETE from conversations')
# print(res)