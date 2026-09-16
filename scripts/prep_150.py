import pandas as pd
import json

df = pd.read_csv('golden_set_unlabelled.csv')
df = df.head(150) # Just take 150

queries = []
for idx, row in df.iterrows():
    queries.append({
        "id": row['tweet_id_cust'],
        "customer_text": row['text_cust'],
        "brand_response": row['text_brand']
    })

with open('golden_150.json', 'w', encoding='utf-8') as f:
    json.dump(queries, f, indent=2)

print("Saved golden_150.json")
