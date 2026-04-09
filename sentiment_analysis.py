import torch
import re
import os
import sys
import json
import csv
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 1. Load Model & Tokenizer ─────────────────────────────────
MODEL_PATH = "./sentiment_model"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ── 2. Read Transcript File ───────────────────────────────────
TRANSCRIPT_PATH = "transcript.txt"

if len(sys.argv) > 1:
    TRANSCRIPT_PATH = sys.argv[1]

if not os.path.exists(TRANSCRIPT_PATH):
    print(f"File not found: {TRANSCRIPT_PATH}")
    sys.exit(1)

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript_text = f.read()

print(f"Loaded: {TRANSCRIPT_PATH}")

# ── 3. Parse Every Row ────────────────────────────────────────
def parse_transcript(text):
    entries = []
    pattern = r'\[(\d{2}:\d{2}:\d{2})\] \[(YOU|CLIENT)\] (.+)'
    current = None
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('==='):
            continue
        match = re.match(pattern, line)
        if match:
            if current:
                entries.append(current)
            ts, speaker, content = match.groups()
            current = {'timestamp': ts, 'speaker': speaker, 'text': content.strip()}
        else:
            if current:
                current['text'] += ' ' + line
    if current:
        entries.append(current)
    return entries

entries = parse_transcript(transcript_text)
print(f"\nTotal rows: {len(entries)}\n")

# ── 4. Predict Function ───────────────────────────────────────
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label_emoji = {
    "negative": "😠 Negative",
    "neutral" : "😐 Neutral",
    "positive": "😊 Positive",
}

def predict(text):
    if not text.strip():
        return "😐 Neutral", 0.0, 1

    inputs = tokenizer(
        text,
        return_tensors = "pt",
        truncation     = True,
        padding        = "max_length",
        max_length     = 128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=-1)
    label_id   = torch.argmax(probs).item()
    confidence = round(probs[0][label_id].item() * 100, 1)
    label      = id2label[label_id]
    return label_emoji[label], confidence, label_id

# ── 5. Run on Every Row ───────────────────────────────────────
results = []
for i, entry in enumerate(entries, 1):
    sentiment, confidence, label_id = predict(entry["text"])
    results.append({
        "row"       : i,
        **entry,
        "sentiment" : sentiment,
        "confidence": confidence,
        "label_id"  : label_id
    })

# ── 6. Print Sentiment for Every Row ─────────────────────────
print("="*100)
print(f"  {'Row':<5} {'Time':<12} {'Speaker':<10} {'Sentiment':<20} {'Conf%':<8} Text")
print("="*100)
for r in results:
    preview = r['text'][:48] + ("..." if len(r['text']) > 48 else "")
    print(f"  {r['row']:<5} [{r['timestamp']}]  [{r['speaker']:<7}]  {r['sentiment']:<20} {r['confidence']:<8} {preview}")

# ── 7. Average Confidence per Sentiment ──────────────────────
print("\n" + "="*55)
print("Average Confidence per Sentiment")
print("="*55)
sentiment_scores = {}
for r in results:
    s = r["sentiment"]
    if s not in sentiment_scores:
        sentiment_scores[s] = []
    sentiment_scores[s].append(r["confidence"])

avg_confidence = {}
for sentiment, scores in sentiment_scores.items():
    avg = round(sum(scores) / len(scores), 1)
    avg_confidence[sentiment] = avg
    print(f"  {sentiment:<22} avg confidence: {avg}%")

# ── 8. Count & Dominant Sentiment ────────────────────────────
print("\n" + "="*55)
print("🏆 Which Sentiment Appears Most")
print("="*55)
all_sentiments = [r["sentiment"] for r in results]
counts         = Counter(all_sentiments)
total          = len(all_sentiments)
dominant       = counts.most_common(1)[0][0]

for sentiment, count in counts.most_common():
    pct = round(count / total * 100)
    bar = "█" * int(pct // 5)
    print(f"  {sentiment:<22} {count} row(s)  {bar} {pct}%")

print(f"\n  Dominant sentiment : {dominant}")
print(f"  Total rows analyzed: {total}")

# ── 9. Build Output Paths ─────────────────────────────────────
transcript_dir  = os.path.dirname(os.path.abspath(TRANSCRIPT_PATH))
transcript_name = os.path.splitext(os.path.basename(TRANSCRIPT_PATH))[0]
timestamp_now   = datetime.now().strftime("%Y%m%d_%H%M%S")

json_path = os.path.join(transcript_dir, f"{transcript_name}_sentiment.json")
csv_path  = os.path.join(transcript_dir, f"{transcript_name}_sentiment.csv")
txt_path  = os.path.join(transcript_dir, f"{transcript_name}_sentiment.txt")


json_output = {
    "meta": {
        "transcript_file"  : TRANSCRIPT_PATH,
        "analyzed_at"      : datetime.now().isoformat(),
        "total_rows"       : total,
        "dominant_sentiment: ": dominant.replace("😠 ", "").replace("😐 ", "").replace("😊 ", ""),
    },
    "summary": {
        sentiment.replace("😠 ", "").replace("😐 ", "").replace("😊 ", ""): {
            "count"          : count,
            "percentage"     : round(count / total * 100, 1),
            "avg_confidence" : avg_confidence.get(sentiment, 0)
        }
        for sentiment, count in counts.items()
    },
    "rows": [
        {
            "row"        : r["row"],
            "timestamp"  : r["timestamp"],
            "speaker"    : r["speaker"],
            "text"       : r["text"],
            "sentiment"  : r["sentiment"].replace("😠 ", "").replace("😐 ", "").replace("😊 ", ""),
            "confidence" : r["confidence"]
        }
        for r in results
    ]
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=4, ensure_ascii=False)

print(f"\nJSON saved : {json_path}")

# ── 11. Save as CSV (for CRM bulk import — Excel, Zoho, HubSpot)
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["row", "timestamp", "speaker", "text", "sentiment", "confidence"])
    for r in results:
        writer.writerow([
            r["row"],
            r["timestamp"],
            r["speaker"],
            r["text"],
            r["sentiment"].replace("😠 ", "").replace("😐 ", "").replace("😊 ", ""),
            r["confidence"]
        ])

print(f"CSV saved  : {csv_path}")

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(f"SENTIMENT ANALYSIS REPORT\n")
    f.write(f"Transcript : {TRANSCRIPT_PATH}\n")
    f.write(f"Analyzed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*100 + "\n")
    f.write(f"  {'Row':<5} {'Time':<12} {'Speaker':<10} {'Sentiment':<20} {'Conf%':<8} Text\n")
    f.write("="*100 + "\n")
    for r in results:
        preview = r['text'][:48] + ("..." if len(r['text']) > 48 else "")
        f.write(f"  {r['row']:<5} [{r['timestamp']}]  [{r['speaker']:<7}]  {r['sentiment']:<20} {r['confidence']:<8} {preview}\n")
    f.write("\n" + "="*55 + "\n")
    f.write("Average Confidence per Sentiment\n")
    f.write("="*55 + "\n")
    for sentiment, avg in avg_confidence.items():
        f.write(f"  {sentiment:<22} avg confidence: {avg}%\n")
    f.write("\n" + "="*55 + "\n")
    f.write("Which Sentiment Appears Most\n")
    f.write("="*55 + "\n")
    for sentiment, count in counts.most_common():
        pct = round(count / total * 100)
        f.write(f"  {sentiment:<22} {count} row(s)  {pct}%\n")
    f.write(f"\n  Dominant sentiment : {dominant}\n")
    f.write(f"  Total rows analyzed: {total}\n")

print(f"TXT saved  : {txt_path}")

print("File Saved Succesfully")