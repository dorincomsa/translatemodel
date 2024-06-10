import json

# Expense{
#     name: string
#     category: ["supermarket", "restaurant", "subscripiton", "service", "transport", "transfer", "withdraw", "fun", "health", "others"]
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
        

    add_entry(data, [
            f"Which month have I spent the most on",
            f"Which month have I spent the most money on",
            f"What is the month I have spent the most money on"
        ],
        f"SELECT month, SUM(value) FROM expenses GROUP BY month ORDER BY value DESC LIMIT 1"
    )        
        
        
    add_entry(data, [
                f"Which category have I spent the most money on",
                f"What is the category I have spent the most money on"
            ],
            f"SELECT category, SUM(value) FROM expenses GROUP BY category ORDER BY value DESC LIMIT 1"
    )
    
    add_entry(data, [
            f"What is my biggest expense",
            f"What have I spent the most money on",
            f"What have I spent the most on",
        ],
        f"SELECT * FROM expenses ORDER BY value DESC LIMIT 1"
    )
    
    add_entry(data, [
            f"What is the total amount spent in <month>", 
            f"What is the total amount spent in month <month>", 
            f"How much money have I spent in <month>"
        ], f"SELECT SUM(value) FROM expenses WHERE month = ' <month> '")
    
    add_entry(data, [
            f"What is the total amount spent in <month> for each category", 
            f"How much money have I spent in <month> for each category"
        ], f"SELECT category , SUM(value) FROM expenses WHERE month = ' <month> ' GROUP BY category")
    

    # add_entry(data, [
    #         f"",   
    #     ],
    #     f"",   
    # )

    add_entry(data, [
        f"How much money have I spent each month",
        f"How much money have I spent each month last year",
    ], f"SELECT month , SUM(value) FROM expenses GROUP BY month")

    return data
        
def create_and_push():

    with open("train_data.json", "w") as out:
        data = []
        for i in range(30):
            set = create_data()
            data.extend(set)

        json.dump(data, out, ensure_ascii=False, indent=4)


def load_custom_data():
    with open("train_data.json") as f:
        data = json.load(f)
        return data
    

# create_and_push()