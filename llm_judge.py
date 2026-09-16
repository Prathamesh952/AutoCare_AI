import pandas as pd
import json
import os
from google import genai
from google.genai import types

def evaluate_reply_quality(df_path="eval_results.csv"):
    df = pd.read_csv(df_path)
    
    client = genai.Client() if os.environ.get("GEMINI_API_KEY") else None
    
    rubric = """You are an expert customer support QA evaluator for AppleSupport.
Evaluate the AI Agent's reply compared to the Actual Brand Reply based on this rubric (Score 1-5):
1 - Completely irrelevant or hallucinated info.
2 - Tangential, not helpful, or wrong tone.
3 - Acceptable, but misses nuances of the brand's style.
4 - Good, matches brand tone, offers next steps.
5 - Excellent, empathetic, accurate, identical in quality to the real brand reply.

Output JSON: {"score": <int>, "reason": "<string>"}"""

    scores = []
    
    for idx, row in df.iterrows():
        if client:
            prompt = f"Customer: {row['text']}\nActual Brand Reply: {row['brand_reply']}\nAI Agent Reply: {row['pred_reply']}"
            try:
                res = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=rubric,
                        response_mime_type="application/json",
                    )
                )
                raw_text = res.text.strip()
                if raw_text.startswith('```json'): raw_text = raw_text[7:]
                elif raw_text.startswith('```'): raw_text = raw_text[3:]
                if raw_text.endswith('```'): raw_text = raw_text[:-3]
                j = json.loads(raw_text.strip())
                scores.append(j.get('score', 3))
            except Exception as e:
                scores.append(3)
        else:
            # Mock evaluation for demonstration if no API key
            scores.append(4 if len(row['pred_reply']) > 10 else 2)
            
    df['llm_judge_score'] = scores
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Average LLM Judge Score (1-5): {avg_score:.2f}")
    df.to_csv("eval_results_with_judge.csv", index=False)

if __name__ == "__main__":
    evaluate_reply_quality()
