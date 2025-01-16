from pathlib import Path
import json
from datasets import load_dataset
from loguru import logger

logger.add(Path(__file__).with_suffix(".log"), mode="w", encoding="utf-8")

dataset = load_dataset("trl-lib/ultrafeedback_binarized")
eval_dataset = dataset["test"]
logger.info(f"{len(eval_dataset) = }")
out_file = 'app/trl/scripts/data/ultrafeedback_binarized.json'
sample_data = []
count = 5
for i, row in enumerate(eval_dataset):
    logger.info(f"{type(row) = }")
    sample_data.append(row)
    if i == count:
        break
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(sample_data, f, ensure_ascii=False, indent=4)

