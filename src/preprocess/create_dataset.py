import os

from src.preprocess.utils import load_file, clean_occupation, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_PATH = "data/raw/"
OCCUPATIONS_PATH = "data/raw/occupations.txt"
PROCESSED_PATH = "data/processed/"

occupations: list[str] = [o.strip() for o in load_file(OCCUPATIONS_PATH)]
logger.info(f"Loaded {len(occupations)} occupations from {OCCUPATIONS_PATH}")

_files: list[list[str]] = [
    [f"{RAW_PATH}/{i}/{f}" for f in os.listdir(os.path.join(RAW_PATH, str(i)))]
    for i in range(1, 5)
]

cleaned_data = []

for anti, pro in _files:
    a_data = load_file(anti)
    p_data = load_file(pro)

    logger.info(f"Anti: {len(a_data)}, Pro: {len(p_data)}")

    for data in zip(a_data, p_data):
        cleaned_anti = clean_occupation(data[0], occupations)
        cleaned_pro = clean_occupation(data[1], occupations)

        new_data = {"anti": cleaned_anti, "pro": cleaned_pro}

        cleaned_data.append(new_data)

save_json(os.path.join(PROCESSED_PATH, "dataset.json"), cleaned_data)
logger.info(f"Saved cleaned dataset with {len(cleaned_data)} entries to {PROCESSED_PATH}/dataset.json")
