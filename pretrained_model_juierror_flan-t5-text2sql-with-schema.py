# Load model directly
import datetime
import nltk.translate.bleu_score as bleu
import numpy as np

from typing import Dict, List
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

model_path = "juierror/flan-t5-text2sql-with-schema"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)



def prepare_input(question: str, table: List[str]):
    table_prefix = "table:"
    question_prefix = "question:"
    join_table = ",".join(table)
    inputs = f"{question_prefix} {question} {table_prefix} {join_table}"
    input_ids = tokenizer(inputs, max_length=512, return_tensors="tf").input_ids
    return input_ids

def inference(question: str, table: List[str]) -> str:
    input_data = prepare_input(question=question, table=table)
    # input_data = input_data.to(model.device)
    outputs = model.generate(inputs=input_data, num_beams=1, top_k=1, max_length=30)
    # outputs = model.generate(inputs=input_data, num_beams=10, top_k=10, max_length=700)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result


# print(inference(question="get people name with age equal 25", table=["id", "name", "age"]))
# print(inference(
#     "What is the total amount spent on supermarket",
#     table= ['name', 'category', 'value', 'day', 'month', 'year']
# ))



def eval(reference_translations, model_translations):
    bleu_scores = []
    for ref, hyp in zip(reference_translations, model_translations):
        bleu_score = bleu.sentence_bleu([ref], hyp)
        bleu_scores.append(bleu_score)

    average_bleu_score = np.mean(bleu_scores)
    return average_bleu_score

from train_data import load_custom_data
import random
data = load_custom_data()
# data = data[:10]
data = random.sample(data, 10)

questions = [x['question'] for x in data]
queries = [x['query'] for x in data]


start_time = datetime.datetime.now()
gen_queries = [inference(q,['name', 'category', 'value', 'day', 'month', 'year']) for q in questions]
gen_queries = [x.replace('table', 'expenses') for x in gen_queries]
stop_time = datetime.datetime.now()

duration = stop_time - start_time
print(duration.total_seconds())

score = eval(queries, gen_queries)
print(score)