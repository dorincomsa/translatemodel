from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import re
import unicodedata
from urllib.parse import urlparse

from model import Seq2Seq
from sql import SQL
from dateutil import parser

model = Seq2Seq()
model.load_model()

sql = SQL()


def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def pre_process(question):
  question = unicode_to_ascii(question.lower().strip())
  question = re.sub(r"([?.!,¿])", r" \1 ", question)
  question = re.sub(r'[" "]+', " ", question)
  question = question.strip()

  months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
  months_replaced = []
  processed_question = []
  for word in question.split(' '):
     if word in months:
      processed_question.append('<month>')
      months_replaced.append(word)
     else:
      processed_question.append(word)

  return ' '.join(processed_question), months_replaced

def post_process(query, values):
  query = query.removeprefix( '<start> ')
  query = query.removesuffix( ' <end>')
  processed_query = []
  for word in query.split(' '):
    if word == '<month>':
       processed_query.append(values[0])
       values = values[1:]
    else:
       processed_query.append(word)

  processed_query = ' '.join(processed_query)
  #print(processed_query)
  pattern = r"' ([^']*) '"
  processed_query = re.sub(pattern, r"'\1'", processed_query)
  return processed_query


class HttpController(BaseHTTPRequestHandler):

  def set_headers(self, status_code=200, content_type='application/json'):
    self.send_response(status_code)
    self.send_header('Content-type', content_type)
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
    self.end_headers()

  def do_OPTIONS(self):
    self.set_headers()

  def do_POST(self):
    content_length = int(self.headers['Content-Length'])
    post_data = self.rfile.read(content_length).decode('utf-8')
    json_data = json.loads(post_data)

    
    if(self.path == '/conversations'):
      
      processed_query = ""
      conversation = None

      try:
        question = json_data.get('question')
        processed_question, values = pre_process(question)
        query = model.translate(processed_question)
        processed_query = post_process(query[0], values)
      except:
        self.set_headers(500)
        self.wfile.write(json.dumps({'error': 'Error generating the query'}).encode('utf-8'))
        return
        
      try:
        response = sql.executeQuery(processed_query)
        response = json.dumps(response)
        conversation = sql.insertConversation(question, response)
      except:
        self.set_headers(500)
        self.wfile.write(json.dumps({
          'error': 'Error executing the query'
        }).encode('utf-8'))
        return

      
      conversation['query'] = processed_query
      self.set_headers()
      self.wfile.write(json.dumps(conversation).encode('utf-8'))

    if(self.path == '/expenses'):
      name = json_data.get('name')
      category = json_data.get('category')
      value = json_data.get('value')
      date = parser.parse(json_data.get('date'))
      year = date.year
      month = date.strftime('%b').lower()
      day = date.day

      expense = sql.insertExpense(name, category, value, year, month, day)

      self.set_headers()
      self.wfile.write(json.dumps(expense).encode('utf-8'))

  def do_GET(self):
    self.set_headers()

    if(self.path == '/conversations'):
      response = sql.getConversations()
      self.wfile.write(json.dumps(response).encode('utf-8'))

      
    if(self.path == '/expenses'):
      response = sql.getExpenses()
      self.wfile.write(json.dumps(response).encode('utf-8'))

  def do_DELETE(self):
    self.set_headers()

    parsed_path = urlparse(self.path)
    path_parts = parsed_path.path.split('/')
    if len(path_parts) == 3 and path_parts[1] == 'expenses':
        try:
            expense_id = path_parts[2]
            sql.deleteExpense(expense_id)
            response = {"status": "success", "message": f"Expense with id {expense_id} deleted."}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            response = {"status": "error", "message": str(e)}
            self.wfile.write(json.dumps(response).encode('utf-8'))
    elif len(path_parts) == 3 and path_parts[1] == 'conversations':
        try:
            conversation_id = path_parts[2]
            sql.deleteConversation(expense_id)
            response = {"status": "success", "message": f"Conversation with id {conversation_id} deleted."}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            response = {"status": "error", "message": str(e)}
            self.wfile.write(json.dumps(response).encode('utf-8'))


host = 'localhost'
port = 8080
httpd = HTTPServer((host, port), HttpController)
print(f'Server stared on port: {port}')
httpd.serve_forever()