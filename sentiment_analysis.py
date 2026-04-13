import torch
import re
import os
import sys
import json
import csv
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 1. Load Model & Tokenizer ─────────────────────────────────────────────────
MODEL_PATH = "./sentiment_model_v2"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ── 2. Read Transcript File ───────────────────────────────────────────────────
TRANSCRIPT_PATH = "transcript1.txt"
if len(sys.argv) > 1:
    TRANSCRIPT_PATH = sys.argv[1]

if not os.path.exists(TRANSCRIPT_PATH):
    print(f"File not found: {TRANSCRIPT_PATH}")
    sys.exit(1)

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript_text = f.read()

print(f"Loaded: {TRANSCRIPT_PATH}")

# ── 3. Detect Language from Header ───────────────────────────────────────────
def detect_lang(text):
    for line in text.split('\n'):
        if line.startswith('=== Language:'):
            return line.split(':')[1].replace('===', '').strip().lower()
    return "english"

transcript_lang = detect_lang(transcript_text)
print(f"Detected language: {transcript_lang}")

# ── 4. Parse Every Row ────────────────────────────────────────────────────────
# Handles both formats:
#   [12:10:29] [CLIENT] text...
#   [12:10:29] [CLIENT] [Hinglish] text...
#   [12:10:29] [CLIENT] [English] text...

def parse_transcript(text):
    entries = []
    pattern = r'\[(\d{2}:\d{2}:\d{2})\] \[(YOU|CLIENT)\] (?:\[\w+\] )?(.+)'
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
print(f"Total rows parsed: {len(entries)}\n")

# ── 5. Hinglish Translation ───────────────────────────────────────────────────
# Install: pip install deep-translator
# If not installed, Hinglish text will still run but accuracy will be lower

def translate_to_english(text):
    """Translate any language to English using Google Translate."""
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated if translated else text
    except ImportError:
        # deep-translator not installed — warn once and skip
        return text
    except Exception:
        return text

IS_HINGLISH = transcript_lang in ["hinglish", "hi", "hindi"]

if IS_HINGLISH:
    print("Hinglish detected — translating lines to English before inference...")
    try:
        from deep_translator import GoogleTranslator
        print("deep-translator available ✓")
    except ImportError:
        print("WARNING: deep-translator not installed. Run: pip install deep-translator")
        print("Continuing without translation (accuracy may be lower for Hinglish)\n")

# ── 6. Label Map & Emoji ──────────────────────────────────────────────────────
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label_emoji = {
    "negative" : "😠 Negative",
    "neutral"  : "😐 Neutral",
    "positive" : "😊 Positive",
}

# ── 7. Predict Function ───────────────────────────────────────────────────────
def predict(text):
    if not text.strip():
        return "😐 Neutral", 0.0, 1

    # Translate Hinglish → English before inference
    input_text = translate_to_english(text) if IS_HINGLISH else text

    inputs = tokenizer(
        input_text,
        return_tensors = "pt",
        truncation     = True,
        padding        = "max_length",
        max_length     = 128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=-1)
    label_id   = torch.argmax(probs).item()
    confidence = round(probs[0][label_id].item() * 100, 1)
    label      = id2label[label_id]
    return label_emoji[label], confidence, label_id

# ── 8. Run Inference on Every Row ─────────────────────────────────────────────
results = []
for i, entry in enumerate(entries, 1):
    sentiment, confidence, label_id = predict(entry["text"])
    results.append({
        "row"        : i,
        **entry,
        "sentiment"  : sentiment,
        "confidence" : confidence,
        "label_id"   : label_id,
    })

# ── 9. Print Table ────────────────────────────────────────────────────────────
print("=" * 105)
print(f"  {'Row':<5} {'Time':<12} {'Speaker':<10} {'Sentiment':<20} {'Conf%':<8} Text")
print("=" * 105)
for r in results:
    preview = r['text'][:50] + ("..." if len(r['text']) > 50 else "")
    print(f"  {r['row']:<5} [{r['timestamp']}]  [{r['speaker']:<7}]  {r['sentiment']:<20} {r['confidence']:<8} {preview}")

# ── 10. Average Confidence per Sentiment ─────────────────────────────────────
print("\n" + "=" * 60)
print("Average Confidence per Sentiment")
print("=" * 60)
sentiment_scores = {}
for r in results:
    sentiment_scores.setdefault(r["sentiment"], []).append(r["confidence"])

avg_confidence = {}
for sentiment, scores in sentiment_scores.items():
    avg = round(sum(scores) / len(scores), 1)
    avg_confidence[sentiment] = avg
    print(f"  {sentiment:<25} avg confidence: {avg}%")

# ── 11. Count & Distribution ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Sentiment Distribution")
print("=" * 60)
all_sentiments = [r["sentiment"] for r in results]
counts         = Counter(all_sentiments)
total          = len(all_sentiments)
dominant       = counts.most_common(1)[0][0]

# Print in fixed order: positive → neutral → negative
for label in ["😊 Positive", "😐 Neutral", "😠 Negative"]:
    count = counts.get(label, 0)
    pct   = round(count / total * 100) if total else 0
    bar   = "█" * int(pct // 5)
    print(f"  {label:<25} {count:>3} row(s)  {bar} {pct}%")

print(f"\n  Dominant sentiment  : {dominant}")
print(f"  Total rows analyzed : {total}")
print(f"  Transcript language : {transcript_lang}")

# ── 12. Build Output Paths ────────────────────────────────────────────────────
transcript_dir  = os.path.dirname(os.path.abspath(TRANSCRIPT_PATH))
transcript_name = os.path.splitext(os.path.basename(TRANSCRIPT_PATH))[0]

json_path = os.path.join(transcript_dir, f"{transcript_name}_sentiment.json")
csv_path  = os.path.join(transcript_dir, f"{transcript_name}_sentiment.csv")
txt_path  = os.path.join(transcript_dir, f"{transcript_name}_sentiment.txt")

def clean_emoji(s):
    return s.replace("😠 ", "").replace("😐 ", "").replace("😊 ", "")

# ── 13. Save JSON ─────────────────────────────────────────────────────────────
json_output = {
    "meta": {
        "transcript_file"   : TRANSCRIPT_PATH,
        "transcript_lang"   : transcript_lang,
        "analyzed_at"       : datetime.now().isoformat(),
        "total_rows"        : total,
        "dominant_sentiment": clean_emoji(dominant),
    },
    "summary": {
        clean_emoji(sentiment): {
            "count"          : count,
            "percentage"     : round(count / total * 100, 1),
            "avg_confidence" : avg_confidence.get(sentiment, 0),
        }
        for sentiment, count in counts.items()
    },
    "rows": [
        {
            "row"        : r["row"],
            "timestamp"  : r["timestamp"],
            "speaker"    : r["speaker"],
            "text"       : r["text"],
            "sentiment"  : clean_emoji(r["sentiment"]),
            "confidence" : r["confidence"],
        }
        for r in results
    ],
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=4, ensure_ascii=False)
print(f"\nJSON saved : {json_path}")

# ── 14. Save CSV ──────────────────────────────────────────────────────────────
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["row", "timestamp", "speaker", "text", "sentiment", "confidence"])
    for r in results:
        writer.writerow([
            r["row"], r["timestamp"], r["speaker"],
            r["text"], clean_emoji(r["sentiment"]), r["confidence"],
        ])
print(f"CSV saved  : {csv_path}")

# ── 15. Save TXT ──────────────────────────────────────────────────────────────
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("SENTIMENT ANALYSIS REPORT\n")
    f.write(f"Transcript  : {TRANSCRIPT_PATH}\n")
    f.write(f"Language    : {transcript_lang}\n")
    f.write(f"Analyzed at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 105 + "\n")
    f.write(f"  {'Row':<5} {'Time':<12} {'Speaker':<10} {'Sentiment':<20} {'Conf%':<8} Text\n")
    f.write("=" * 105 + "\n")
    for r in results:
        preview = r['text'][:50] + ("..." if len(r['text']) > 50 else "")
        f.write(f"  {r['row']:<5} [{r['timestamp']}]  [{r['speaker']:<7}]  {r['sentiment']:<20} {r['confidence']:<8} {preview}\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("Average Confidence per Sentiment\n")
    f.write("=" * 60 + "\n")
    for sentiment, avg in avg_confidence.items():
        f.write(f"  {sentiment:<25} avg confidence: {avg}%\n")
    f.write("\n" + "=" * 60 + "\n")
    f.write("Sentiment Distribution\n")
    f.write("=" * 60 + "\n")
    for label in ["😊 Positive", "😐 Neutral", "😠 Negative"]:
        count = counts.get(label, 0)
        pct   = round(count / total * 100) if total else 0
        f.write(f"  {label:<25} {count:>3} row(s)  {pct}%\n")
    f.write(f"\n  Dominant sentiment  : {dominant}\n")
    f.write(f"  Total rows analyzed : {total}\n")
    f.write(f"  Transcript language : {transcript_lang}\n")

print(f"TXT saved  : {txt_path}")
print("\nDone!")