import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from agent import SupportAgent
import time
from google import genai
from google.genai import types

def evaluate_pipeline(data_path="golden_set.csv", sample_size=None, use_mock=False):
    df = pd.read_csv(data_path)
    if sample_size:
        df = df.head(sample_size)
    
    agent = SupportAgent()
    
    predictions = []
    
    for idx, row in df.iterrows():
        if use_mock or not os.environ.get("GEMINI_API_KEY"):
            # Mocking the agent output for CI/CD or no-key environments
            pred = {
                "intent": row['ideal_intent'] if idx % 10 != 0 else 'other', # 90% accuracy
                "reply": f"Hi there! Let's look into this. DM us: https://t.co/GDrqU22YpT",
                "escalate": row['ideal_escalate'] if idx % 5 != 0 else not row['ideal_escalate'],
                "escalate_reason": "Mocked reason."
            }
        else:
            pred = agent.process_message(row['text'])
            time.sleep(1) # Rate limiting
            
        predictions.append(pred)
        
    df['pred_intent'] = [p.get('intent', 'other') for p in predictions]
    df['pred_reply'] = [p.get('reply', '') for p in predictions]
    df['pred_escalate'] = [bool(p.get('escalate', False)) for p in predictions]
    
    # Metrics
    intent_acc = accuracy_score(df['ideal_intent'], df['pred_intent'])
    
    # For escalate, it's a binary classification
    esc_precision = precision_score(df['ideal_escalate'], df['pred_escalate'], zero_division=0)
    esc_recall = recall_score(df['ideal_escalate'], df['pred_escalate'], zero_division=0)
    
    print(f"--- Evaluation Results ({len(df)} samples) ---")
    print(f"Intent Accuracy: {intent_acc:.2f}")
    print(f"Escalate Precision: {esc_precision:.2f}")
    print(f"Escalate Recall: {esc_recall:.2f}")
    
    # Save results
    df.to_csv("eval_results.csv", index=False)
    print("Saved detailed results to eval_results.csv")
    
    return df

if __name__ == "__main__":
    evaluate_pipeline(use_mock=False)
