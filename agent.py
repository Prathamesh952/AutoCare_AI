import json
import os
from google import genai
from google.genai import types

class SupportAgent:
    def __init__(self, model_name="gemini-2.5-flash"):
        # Relies on GEMINI_API_KEY environment variable
        self.model_name = model_name
        try:
            self.client = genai.Client()
        except ValueError:
            self.client = None
        
        self.system_instruction = """You are an AI customer support agent for AppleSupport on Twitter.
Your job is to read a customer's incoming tweet and output a JSON object with three fields:
1. "intent": Classify into one of: ["ios_update", "battery_power", "hardware_purchase", "bug_feature", "other"]
2. "reply": A short, empathetic drafted reply (under 280 characters) grounded in how AppleSupport resolves issues (e.g., asking for iOS version, offering DM link https://t.co/GDrqU22YpT).
3. "escalate": boolean (true/false) whether to escalate to a human.
4. "escalate_reason": A short reason for the escalate decision.

Escalate if the user is extremely angry, threatening, or if the issue requires checking account/order details or booking a physical store appointment. Otherwise, auto-handle with troubleshooting questions.
"""

    def process_message(self, text):
        prompt = f"Customer tweet:\n{text}\n\nOutput JSON strictly."
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                ),
            )
            # Clean potential markdown formatting
            raw_text = response.text.strip()
            if raw_text.startswith('```json'):
                raw_text = raw_text[7:]
            if raw_text.startswith('```'):
                raw_text = raw_text[3:]
            if raw_text.endswith('```'):
                raw_text = raw_text[:-3]
            
            return json.loads(raw_text.strip())
        except Exception as e:
            # Fallback in case of error
            return {
                "intent": "other",
                "reply": "We'd like to look into this with you. Please DM us here: https://t.co/GDrqU22YpT",
                "escalate": True,
                "escalate_reason": f"Error processing: {str(e)}"
            }

if __name__ == "__main__":
    agent = SupportAgent()
    res = agent.process_message("My battery is dying so fast after the update!")
    print(json.dumps(res, indent=2))
