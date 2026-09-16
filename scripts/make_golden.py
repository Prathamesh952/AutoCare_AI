import pandas as pd
import json
import re

with open('golden_150.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

# Define Intents:
# - ios_update (update, ios 11, upgrade)
# - battery_power (battery, charge, die, drain, power)
# - bug_feature (keyboard, screen, mail, app, crash)
# - hardware_purchase (buy, order, repair, store, appointment)
# - other

def classify_intent(text):
    t = text.lower()
    if any(w in t for w in ['update', 'ios', 'upgrade', '11.1', '11.0']):
        return 'ios_update'
    if any(w in t for w in ['battery', 'charge', 'die', 'drain', 'power', 'plug']):
        return 'battery_power'
    if any(w in t for w in ['buy', 'order', 'repair', 'store', 'appointment', 'broken', 'screen']):
        return 'hardware_purchase'
    if any(w in t for w in ['keyboard', 'mail', 'app', 'crash', 'glitch', 'wifi', 'bluetooth', 'symbol']):
        return 'bug_feature'
    return 'other'

def should_escalate(intent, text):
    # If it's a hardware purchase/repair, usually needs a human to book appointment.
    # If it's an angry customer (swearing), escalate.
    t = text.lower()
    if intent == 'hardware_purchase':
        return True, "Requires store appointment or checking order status."
    if any(w in t for w in ['fuck', 'shit', 'piss', 'annoying', 'ruin']):
        return True, "Customer expresses high frustration."
    return False, "Can be auto-handled with standard troubleshooting steps."

golden_data = []
for q in queries:
    intent = classify_intent(q['customer_text'])
    esc, reason = should_escalate(intent, q['customer_text'])
    golden_data.append({
        'tweet_id': q['id'],
        'text': q['customer_text'],
        'ideal_intent': intent,
        'ideal_escalate': esc,
        'ideal_escalate_reason': reason,
        'brand_reply': q['brand_response']
    })

df_golden = pd.DataFrame(golden_data)
df_golden.to_csv('golden_set.csv', index=False)
print("Saved golden_set.csv")
