import json


# Expense{
#     name: string
#     category: supermarket | restaurant | services | travel  | others;
#     value: number;
#     year: number;
#     month: string;
#     day: number;
# }

def add_entry(data, questions, query):
    for question in questions:
        entry = {}
        entry["question"] = question
        entry["query"] = query
        data.append(entry)

def create_data():
    data = []

    categories = ["supermarket", "restaurant", "travel", "service", "transport", "transfer", "withdraw", "fun", "health", "others"]
    for category in categories:
        add_entry(data, [f"what are the {category} expenses"], f"SELECT * FROM expenses WHERE category = ' {category} '")

        add_entry(data, [
                f"What is the total amount spent on {category}",
                f"How much have I spent on {category}",
                f"How much have been spent on {category}",
                f"How much money have I spent on {category}",
            ], f"SELECT SUM(value) FROM expenses WHERE category =  ' {category} '")
        
        add_entry(data, [f"What is the total amount spent on {category} in <month>"], f"SELECT SUM(value) FROM expenses WHERE category = ' {category} ' AND month =  ' <month> '")
        
        # pe ce am cheltuit cel mai mult luna aceasta 
        # pe ce am cheltuit cel mai putin luna aceasta 
    
    add_entry(data, [
            f"What is the total amount spent in <month>", 
            f"How much money have I spent in <month>"
        ], f"SELECT SUM(value) FROM expenses WHERE month = ' <month> '")
    
    add_entry(data, [
            f"What is the total amount spent in <month> for each category", 
            f"How much money have I spent in <month> for each category"
        ], f"SELECT category , SUM(value) FROM expenses WHERE month = ' <month> ' GROUP BY category")
    
    add_entry(data, [
        f"How much money have I spent each month last year"
    ], f"SELECT month , SUM(value) FROM expenses GROUP BY month")

    return data
        
def create_and_push():

    with open("custom_data.json", "w") as out:
        data = []

        for i in range(30):
            set = create_data()
            data.extend(set)

        json.dump(data, out, ensure_ascii=False, indent=4)


def load_custom_data():
    with open("custom_data.json") as f:
        data = json.load(f)
        return data
    

# create_and_push()