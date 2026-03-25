from datasets import load_dataset

train_dataset = load_dataset("LabHC/bias_in_bios", split='train')

print(train_dataset[1])
