import re
import json


def load_file(file_path: str):
    with open(file_path, "r") as file:
        return file.readlines()

def save_json(file_path: str, data: list[dict]):
    with open(file_path, "w") as file:
        json.dump(data, file)

def clean_occupation(data: str, occupations_list: list[str]) -> str:
    """
    Cleans a string by removing leading numbers and brackets around occupations.
    e.g. "1 [The developer] argued with [her]." -> "The developer argued with [her]."
    """
    cleaned_data = re.sub(r"^\d+\s+", "", data) # removing numbers in the beginning

    occupation_pattern = "|".join(re.escape(occ) for occ in occupations_list)

    def replace_brackets(match):
        phrase = match.group(1)
        if re.search(occupation_pattern, phrase):
            return phrase
        return match.group(0)

    cleaned_data = re.sub(r"\[(.*?)\]", replace_brackets, cleaned_data)
    return cleaned_data.strip()