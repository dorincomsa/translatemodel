import datetime
import unittest
import sqlite3
from unittest.mock import patch
from sql import SQL 

class TestSQL(unittest.TestCase):

    def setUp(self):
        self.sql = SQL()
        self.sql.connection = sqlite3.connect(":memory:")
        self.sql.cursour = self.sql.connection.cursor()
        self.sql.createTables()

    def tearDown(self):
        self.sql.connection.close()

    def test_createTables(self):
        self.sql.createTables()
        self.sql.cursour.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='expenses'")
        self.assertIsNotNone(self.sql.cursour.fetchone())
        self.sql.cursour.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
        self.assertIsNotNone(self.sql.cursour.fetchone())

    def test_dropTables(self):
        self.sql.dropTables()
        self.sql.cursour.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='expenses'")
        self.assertIsNone(self.sql.cursour.fetchone())
        self.sql.cursour.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
        self.assertIsNone(self.sql.cursour.fetchone())

    def test_insertExpense(self):
        with patch('sql.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2023, 1, 1, 0, 0)
            expense = self.sql.insertExpense('Test Expense', 'Test Category', 100.0, 2023, 1, 1)
        self.assertEqual(expense['name'], 'Test Expense')
        self.assertEqual(expense['category'], 'Test Category')
        self.assertEqual(expense['value'], 100.0)
        self.assertEqual(expense['year'], 2023)
        self.assertEqual(expense['month'], 1)
        self.assertEqual(expense['day'], 1)

    def test_insertConversation(self):
        with patch('sql.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2023, 1, 1, 0, 0)
            conversation = self.sql.insertConversation('Test Question', 'Test Response')
        self.assertEqual(conversation['question'], 'Test Question')
        self.assertEqual(conversation['response'], 'Test Response')
        self.assertEqual(conversation['date'], '2023-01-01T00:00:00')

    def test_getExpenses(self):
        with patch('sql.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2023, 1, 1, 0, 0)
            self.sql.insertExpense('Test Expense', 'Test Category', 100.0, 2023, 1, 1)
        expenses = self.sql.getExpenses()
        self.assertEqual(len(expenses), 1)
        self.assertEqual(expenses[0]['name'], 'Test Expense')

    def test_getConversations(self):
        with patch('sql.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2023, 1, 1, 0, 0)
            self.sql.insertConversation('Test Question', 'Test Response')
        conversations = self.sql.getConversations()
        self.assertEqual(len(conversations), 1)
        self.assertEqual(conversations[0]['question'], 'Test Question')

    def test_deleteExpense(self):
        with patch('sql.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2023, 1, 1, 0, 0)
            expense = self.sql.insertExpense('Test Expense', 'Test Category', 100.0, 2023, 1, 1)
        self.sql.deleteExpense(expense['id'])
        expenses = self.sql.getExpenses()
        self.assertEqual(len(expenses), 0)

    def test_deleteConversation(self):
        with patch('sql.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2023, 1, 1, 0, 0)
            conversation = self.sql.insertConversation('Test Question', 'Test Response')
        self.sql.deleteConversation(conversation['id'])
        conversations = self.sql.getConversations()
        self.assertEqual(len(conversations), 0)

    def test_executeQuery(self):
        self.sql.cursour.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        self.sql.cursour.execute("INSERT INTO test (value) VALUES ('test value')")
        result = self.sql.executeQuery("SELECT value FROM test WHERE id = 1")
        self.assertEqual(result, 'test value')

if __name__ == '__main__':
    unittest.main()