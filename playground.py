import datetime
from model import Seq2seq


model = Seq2seq()
model.load_model()

# model.evaluate_model()

from train_data import load_custom_data
import random
data = load_custom_data()
data = random.sample(data, 10)
questions = [x['question'] for x in data]
queries = [x['query'] for x in data]


start_time = datetime.datetime.now()
gen_queries = [model.translate(q) for q in questions]
stop_time = datetime.datetime.now()

duration = stop_time - start_time
print(duration.total_seconds())