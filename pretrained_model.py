# Load model directly
import nltk.translate.bleu_score as bleu
import numpy as np

from typing import Dict, List
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

model_path = 'juierror/flan-t5-text2sql-with-schema-v2'
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def get_prompt(tables, question):
    prompt = f"""convert question and table into SQL query. tables: {tables}. question: {question}"""
    return prompt

def prepare_input(question: str, tables: Dict[str, List[str]]):
    tables = [f"""{table_name}({",".join(tables[table_name])})""" for table_name in tables]
    tables = ", ".join(tables)
    prompt = get_prompt(tables, question)
    input_ids = tokenizer(prompt, max_length=512, return_tensors="tf").input_ids
    return input_ids

def inference(question: str, tables: Dict[str, List[str]]) -> str:
    input_data = prepare_input(question=question, tables=tables)
    # input_data = input_data.to(model.device)
    outputs = model.generate(inputs=input_data, num_beams=10, top_k=10, max_length=512)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result



print(inference("What is the total amount spent on supermarket",
      {
          "expenses": ['name', 'category', 'value', 'day', 'month', 'year']
      }))



# def eval(reference_translations, model_translations):
#     bleu_scores = []
#     for ref, hyp in zip(reference_translations, model_translations):
#         bleu_score = bleu.sentence_bleu([ref], hyp)
#         bleu_scores.append(bleu_score)

#     average_bleu_score = np.mean(bleu_scores)

# from custom_data import load_custom_data

# data = load_custom_data()
# data = data[:10]

# questions = [x['question'] for x in data]
# queries = [x['query'] for x in data]
# print(questions)
# # gen_queries = [inference(q,['name', 'category', 'value', 'day', 'month', 'year']) for q in questions]
# q= questions[0]
# gen_queries = inference(q,['name', 'category', 'value', 'day', 'month', 'year'])

# score = eval(queries, gen_queries)
# print(score)
# print('red')
# print('red')