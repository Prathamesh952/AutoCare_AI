import pandas as pd
import json

df = pd.read_csv('twcs/twcs.csv')

# Find all tweets by AppleSupport
apple_replies = df[df['author_id'] == 'AppleSupport']

# We need the tweets that AppleSupport replied to
customer_tweet_ids = apple_replies['in_response_to_tweet_id'].dropna().astype(int)

# Get the customer tweets
customer_tweets = df[df['tweet_id'].isin(customer_tweet_ids)]

# Merge to form pairs: Customer Tweet -> AppleSupport Reply
pairs = pd.merge(customer_tweets, apple_replies, left_on='tweet_id', right_on='in_response_to_tweet_id', suffixes=('_cust', '_brand'))

print(f"Total pairs: {len(pairs)}")

# Save a sample of pairs to look at
pairs = pairs[['text_cust', 'text_brand', 'tweet_id_cust', 'tweet_id_brand']]
pairs.to_csv('apple_support_pairs.csv', index=False)

print("Saved apple_support_pairs.csv")
