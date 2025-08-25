from datasets import load_dataset

ds = load_dataset("LabHC/bias_in_bios")

print(ds["train"][1])
