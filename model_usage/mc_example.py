# Multiple Choice Example 
from transformers import AutoTokenizer, DistilBertForMultipleChoice
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-cased")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
print(f"{labels=}")

encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors="pt", padding=True)
outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1
print(f"{encoding=}")
print(f"\n{outputs=}")
# the linear classifier still needs to be trained
loss = outputs.loss
logits = outputs.logits
print(f"\n{loss=}")
print(f"{logits=}")