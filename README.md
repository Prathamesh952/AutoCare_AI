# AutoCare AI: Intelligent Social Media Triage Pipeline

An automated, LLM-powered customer support pipeline built on the [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). This repository focuses specifically on the `@AppleSupport` brand to demonstrate a highly structured, scalable approach to ticket triage, intent classification, and automated reply generation.

---

## 🚀 Quickstart & Reproducibility (Under 15 Minutes)

### Prerequisites
- Python 3.9+
- A Google Gemini API Key. (You can generate one for free at [Google AI Studio](https://aistudio.google.com/)). 

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Prathamesh952/AutoCare_AI.git
   cd AutoCare_AI
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Set your API key in your terminal.
   - On Windows (PowerShell): `$env:GEMINI_API_KEY="your-api-key"`
   - On Mac/Linux: `export GEMINI_API_KEY="your-api-key"`

4. **Run the Evaluation Pipeline:**
   The pipeline will read the golden dataset, process the intents using the AI Agent, and calculate precision, recall, and accuracy.
   ```bash
   python eval_harness.py
   ```

5. **Run the LLM-as-Judge Evaluator:**
   This step utilizes a secondary LLM prompt to score the generated replies against actual AppleSupport historical replies on a 1-5 rubric.
   ```bash
   python llm_judge.py
   ```
   *(Note: If the `GEMINI_API_KEY` is not found, the harness defaults to a deterministic local mock mode so reviewers can still execute the code and view the data flow without API access).*

---

## 🛠 Tech Stack
- **Language:** Python 3.11
- **LLM SDK:** `google-genai` (Gemini 2.5 Flash for high-throughput triage)
- **Data Manipulation:** `pandas`
- **Metrics:** `scikit-learn` (Accuracy, Precision, Recall)

---

## 📖 1. Problem Framing & Scope

### What "Good" Means for AppleSupport
In the context of `@AppleSupport` on Twitter, a "good" AI agent must exhibit three core competencies:
1. **Accurate Triage (Intent):** Quickly identify if a customer is facing a software glitch (`ios_update`), battery failure (`battery_power`), hardware damage (`hardware_purchase`), or a distinct app bug (`bug_feature`).
2. **Safety-First Escalation:** Frustrated customers, hardware repairs, and billing issues must be escalated to humans immediately. A false positive (escalating an easy ticket) is vastly preferable to a false negative (auto-handling a furious customer).
3. **Brand Voice:** Replies must mimic Apple's concise, empathetic, and highly structured format (e.g., acknowledging the issue, asking for iOS versions, and providing standard DM routing links).

### What I Chose *Not* to Build
- **Multi-turn Conversation Memory:** The agent evaluates strictly the *root* (initial) customer tweet to fire the first response. Multi-turn context tracking requires a complex state machine and coreference resolution that introduces massive latency. The highest ROI for an initial deployment is perfect zero-shot triage on the first inbound message.
- **RAG for Documentation:** I did not scrape Apple Support documentation to generate technical step-by-step fixes. Because Twitter's character limit restricts lengthy tutorials, Apple historically prefers routing complex technical issues to DMs.

---

## 📊 2. Golden Set & Evaluation Harness

### The Golden Evaluation Set
The repository includes `golden_set.csv`, a dataset of 150 meticulously curated customer queries. 
- **Sampling & Labelling:** I extracted all root tweets directed at `@AppleSupport` from the Kaggle dataset. I randomly sampled 150 queries and utilized a hybrid approach to label them: initial broad categorization via keyword heuristics (see `scripts/make_golden.py`), followed by manual review to assign the ground-truth `ideal_intent` and `ideal_escalate` decisions.

### Results vs. Baselines

I compared the Gemini 2.5 Flash agent against two baselines:
1. **Trivial Baseline:** Always predicts the majority class ("other"), always auto-handles (never escalates), and replies with a hardcoded static string.
2. **Simple Baseline:** Uses RegEx keyword matching (e.g., if text contains "battery" -> predict `battery_power`) and template-based replies.

| Metric | Trivial Baseline | Simple Baseline (RegEx) | AutoCare AI Agent (Gemini) |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 35% | 68% | **91%** |
| **Escalate Precision** | 0% | 25% | **41%** |
| **Escalate Recall** | 0% | 50% | **82%** |
| **Avg Reply Quality (1-5)** | 1.0 | 2.5 | **4.0** |

*Note: Escalate Precision is intentionally lower (41%) because the system prompt is tuned to aggressively escalate ambiguous or slightly frustrated queries to protect brand reputation, resulting in safe false positives.*

---

## ⚠️ 3. Failure Analysis

Through qualitative review of the agent's output on the dataset, I identified the top 5 recurring failure modes:

1. **Failure Mode: Undetected Sarcasm / Implicit Frustration**
   - *Example:* "Thanks Apple for making my phone totally unusable after the update. Great job."
   - *Hypothesis:* The LLM detects the keywords regarding an iOS update and classifies the intent correctly, but fails to parse the sarcastic tone, resulting in `escalate=False` when it should go to a human.
2. **Failure Mode: Multi-Intent Interference**
   - *Example:* "My screen is cracked and since the update my battery dies in 2 hours."
   - *Hypothesis:* The LLM picks `ios_update` and ignores the `hardware_purchase` (cracked screen), missing the crucial need for a physical Apple Store appointment.
3. **Failure Mode: Overly Cautious Escalation (Hallucinated Risk)**
   - *Example:* "I want to buy the new iPhone X but the site is down, any idea when it's up?"
   - *Hypothesis:* The agent hallucinates that any mention of "buy", "store", or "down" indicates a critical enterprise failure, escalating a simple inquiry that could be auto-handled with a status page link.
4. **Failure Mode: Generic Drafting over Specific Fixes**
   - *Example:* "My mail app keeps typing A instead of I." (A famous iOS 11 bug).
   - *Hypothesis:* The agent drafts a generic "Let's look into this in DM" instead of recognizing the specific, historically documented iOS 11.1 bug and offering the exact fix, missing an opportunity for zero-touch resolution.
5. **Failure Mode: Link Hallucination Loss**
   - *Example:* Agent drafts: "Please DM us at https://t.co/fakeLink"
   - *Hypothesis:* Without strict system-prompt enforcements, the LLM sometimes fabricates Twitter DM shortlinks rather than preserving the exact URL Apple uses (`https://t.co/GDrqU22YpT`).

---

## 🛑 4. What is misleading about my headline number? (Mandatory Section)

While a **91% Intent Accuracy** and a **4.0/5.0 Reply Quality** look phenomenal on paper, these metrics are highly misleading for a production environment for several reasons:

1. **Survivorship & Clean Data Bias:** The `golden_set.csv` of 150 items was pre-filtered for English language and relatively coherent sentence structures. Real-world Twitter firehoses contain a massive long-tail of spam, single-word complaints ("fix it"), link-only tweets, and images without alt-text. On a completely raw production stream, accuracy would likely drop to ~70-75% due to Out-Of-Distribution (OOD) noise.
2. **LLM-as-Judge Bias:** The 4.0 Reply Quality score is graded by an LLM evaluating its own species of generated text. Generative models naturally harbor a stylistic bias toward LLM-written text, artificially inflating the score compared to a blind human A/B test.
3. **Static Temporal Context:** The historical dataset covers issues from ~2017 (e.g., iPhone X, iOS 11). The model achieves high accuracy because these historical issues are heavily present in its base pre-training data. If deployed tomorrow on iOS 18 bugs, performance would drop until the context window is updated via RAG.

---

## 🔮 5. What I'd do next with one more week

1. **RAG Integration for Known Outages:** I would implement a lightweight vector database (like Chroma or FAISS) loaded with current Apple Support articles. Instead of generic DM requests, the agent could perform semantic search to provide exact troubleshooting steps (e.g., "Reset Network Settings").
2. **Confidence Thresholding:** I'd modify the agent to return a probability confidence score. Tickets scoring below 85% confidence would automatically bypass the auto-reply system and route to human QA.
3. **Streamlit Human-in-the-Loop Dashboard:** Build a simple frontend where human agents can view a feed of drafted replies and click "Approve" or "Edit", allowing the system to collect RLHF (Reinforcement Learning from Human Feedback) data to fine-tune a smaller, cheaper model.

---

## 🧠 6. Decision Log (Non-Obvious Engineering Decisions)

1. **Targeted AppleSupport:** Decided to filter the 3M row dataset to a single brand. Apple has highly structured hardware/software intents, making classification more objective than airlines or retail brands where sentiment is heavily blurred.
2. **Ignored Mid-Thread Context:** Decided to process only "root" tweets (where `in_response_to_tweet_id` is NaN). Coreference resolution in multi-turn Twitter threads is extremely noisy; focusing on first-touch triage provides the highest business value.
3. **Collapsed Taxonomy (4 Intents):** Resisted the urge to create 20 micro-intents. Grouping into broad buckets (`ios_update`, `battery_power`, `hardware`, `bug_feature`) prevents LLM confusion and overlapping category distributions.
4. **Unified JSON Payload:** Forced the LLM to output Intent, Reply, Escalate, and Reason all in a single JSON schema. This cuts API latency by 60% compared to chaining three separate sequential prompts.
5. **Chain-of-Thought Enforcement:** Placed the `escalate_reason` field *before* or alongside the boolean `escalate` field to force the LLM to "think out loud", stabilizing the classification accuracy.
6. **Aggressive Escalate Tuning:** Tuned the system prompt to prefer False Positives (escalating unnecessarily) over False Negatives. In customer service, an ignored critical issue causes churn, while a slightly over-escalated queue just costs pennies in agent time.
7. **LLM-as-Judge Setup:** Chose an LLM grading rubric over ROUGE or BLEU scores. Traditional NLP metrics penalize valid empathetic rephrasings, whereas an LLM-judge evaluates semantic helpfulness and tone.
8. **Graceful CI/CD Mocking:** Implemented a `use_mock` fallback in `eval_harness.py`. If a recruiter clones this and doesn't have an API key, the script still runs deterministically, avoiding a crash.
9. **Heuristic Golden Set Bootstrapping:** Rather than hand-typing 150 labels from scratch, I wrote a RegEx script to auto-label the sampled data, then manually spot-checked/corrected the CSV. This mimics how MLOps teams quickly bootstrap datasets.
10. **Hardcoded Brand URLs:** Intentionally injected Apple's standard DM link `https://t.co/GDrqU22YpT` directly into the system prompt to anchor the generation and prevent URL hallucination.
11. **Model Selection (Flash over Pro):** Elected to use `gemini-2.5-flash`. Twitter bots demand high throughput and low latency. The "Pro" models are overkill for text routing and cost significantly more at scale.
12. **Ignored Image/Media Attachments:** Stripped out image processing. While customers upload screenshots of bugs, multimodal OCR slows down inference too much for a V1 MVP. 
