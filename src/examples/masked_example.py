from transformers import AutoTokenizer, DistilBertForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
print(f"{inputs=}")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of <mask>
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
print(f"{mask_token_index=}\n")

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

topk = torch.topk(logits[0, mask_token_index], k=5, dim=-1)
for idx in topk.indices[0]:
    print(tokenizer.decode(idx.item()))

print(f"\n{predicted_token_id=}")
print(f"{tokenizer.decode(predicted_token_id)=}\n")

labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
print(f"{labels=}")
# mask labels of non-<mask> tokens
labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
print(f"{labels=}")

outputs = model(**inputs, labels=labels)
print(f"{round(outputs.loss.item(), 2)=}")