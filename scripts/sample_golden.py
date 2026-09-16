import pandas as pd
import json
import random

# Load original dataset to get the full info
df = pd.read_csv('twcs/twcs.csv')

# Apple support replies
apple_replies = df[df['author_id'] == 'AppleSupport']

# Customer tweet IDs that Apple replied to
customer_tweet_ids = apple_replies['in_response_to_tweet_id'].dropna().astype(int)

# Filter customer tweets that are root tweets (in_response_to_tweet_id is NaN)
customer_tweets = df[df['tweet_id'].isin(customer_tweet_ids)]
root_customer_tweets = customer_tweets[customer_tweets['in_response_to_tweet_id'].isna()]

# Merge pairs
pairs = pd.merge(root_customer_tweets, apple_replies, left_on='tweet_id', right_on='in_response_to_tweet_id', suffixes=('_cust', '_brand'))

print(f"Total root pairs: {len(pairs)}")

# Sample 200
sampled_pairs = pairs.sample(n=200, random_state=42)

# Keep relevant columns
sampled_pairs = sampled_pairs[['tweet_id_cust', 'text_cust', 'text_brand']]
sampled_pairs.to_csv('golden_set_unlabelled.csv', index=False)
print("Saved golden_set_unlabelled.csv")
