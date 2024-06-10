import datetime
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import nltk.translate.bleu_score as bleu
import numpy as np

model_path = 'gaussalgo/T5-LM-Large-text2sql-spider'
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


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
data = random.sample(data, 10)


def runOne(question):
    schema = """
    "expenses" "name" text , "category" text , "value" int , "year" int , "month" text , "day" int
    """

    input_text = " ".join(["Question: ",question, "Schema:", schema])

    model_inputs = tokenizer(input_text, return_tensors="tf")
    outputs = model.generate(**model_inputs, max_length=512)
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_text

import json
with open('results/gaussalgo.json', 'w') as f:
    json.dump(data, f)
    

    questions = [x['question'] for x in data]
    queries = [x['query'] for x in data]


    start_time = datetime.datetime.now()

    model_queries = [runOne(q)[0] for q in questions]
    stop_time = datetime.datetime.now()

    duration = stop_time - start_time
    print(duration.total_seconds())



    res = []
    for i in range(len(model_queries)):
        res.append({
            "question": questions[i],
            "query": queries[i],
            "model_querie": model_queries[i]
        })

    json.dump(res, f, ensure_ascii=False, indent=4)

    blue_score = eval(queries, model_queries)
    print(blue_score)