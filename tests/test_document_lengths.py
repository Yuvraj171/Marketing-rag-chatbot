import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.ingest_documents import load_all_pdfs


documents = load_all_pdfs()

lengths = [len(doc.page_content) for doc in documents]

print("\nDocument length statistics")
print("--------------------------")

print("Total documents:", len(lengths))
print("Min length:", min(lengths))
print("Max length:", max(lengths))
print("Average length:", sum(lengths) // len(lengths))

sorted_lengths = sorted(lengths)

print("\nSample lengths:")
print(sorted_lengths[:10])
print(sorted_lengths[-10:])

buckets = {
    "0-499": 0,
    "500-999": 0,
    "1000-1999": 0,
    "2000-3999": 0,
    "4000+": 0,
}

for length in lengths:
    if length < 500:
        buckets["0-499"] += 1
    elif length < 1000:
        buckets["500-999"] += 1
    elif length < 2000:
        buckets["1000-1999"] += 1
    elif length < 4000:
        buckets["2000-3999"] += 1
    else:
        buckets["4000+"] += 1

print("\nLength buckets:")
for bucket, count in buckets.items():
    print(f"{bucket}: {count}")

# -------------------------------
# Visualization
# -------------------------------

labels = list(buckets.keys())
values = list(buckets.values())

plt.figure(figsize=(8,5))
plt.bar(labels, values)

plt.title("Document Length Distribution")
plt.xlabel("Character Length Range")
plt.ylabel("Number of Documents")

plt.tight_layout()
plt.show()