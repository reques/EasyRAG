"""Run INSIDE the backend container: validate the golden set with the REAL importer."""
import json
import sys

from backend.services.evaluation_import import parse_dataset_import

path = sys.argv[1]
with open(path, "rb") as handle:
    content = handle.read()

parsed = parse_dataset_import("cloudway24_golden_set.json", content)
print("name       :", parsed["name"])
print("description:", parsed["description"][:90], "...")
print("cases      :", len(parsed["cases"]))
print("errors     :", len(parsed["errors"]))
for err in parsed["errors"][:20]:
    print("   ", err)

sample = parsed["cases"][0]
print("\nfirst parsed case (as the importer sees it):")
print(json.dumps({k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                  for k, v in sample.items()}, ensure_ascii=False, indent=2))

# Bucket summary of what the importer extracted
empty_q = sum(1 for c in parsed["cases"] if not c["question"])
no_file = sum(1 for c in parsed["cases"] if not c["expected_file_id"] and not c["expected_filename"])
no_chunks = sum(1 for c in parsed["cases"] if not c["expected_chunk_ids"])
neg = sum(1 for c in parsed["cases"] if c["expect_miss"])
print("\nempty question:", empty_q)
print("no file reference:", no_file)
print("cases with no chunk ids:", no_chunks)
print("expect_miss=true:", neg)
print("\nRESULT:", "PASS - importer accepts every case" if not parsed["errors"] else "FAIL")
sys.exit(0 if not parsed["errors"] else 1)
