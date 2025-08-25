import json
import os
import random
import torch
from transformers import AutoTokenizer, DistilBertForMultipleChoice

# Load once outside function (so it's not reloaded every call)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-cased")

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

class Bias:
    def __init__(self, bias_type , description = ""):
        self.path = f"bencmarking/Steroset_Biases/{self.bias_type}_bias.json"
        if os.path.exists(self.path):
            self.load_json()
            return

        self.bias_type = bias_type
        self.description = description
        self.intra_sentence = []
        self.inter_sentence = []

    def to_dict(self):
        return {
            'bias_type': self.bias_type,
            'description': self.description,
            'intra-sentence': self.intra_sentence,
            'inter-sentence': self.inter_sentence
        }

    def add_sentence(self, sentence, sentence_type):
        if sentence_type == "intra":
            self.intra_sentence.append(sentence)
        elif sentence_type == "inter":
            self.inter_sentence.append(sentence)
    
    def write_json(self):
        save_json(f"bencmarking/Steroset_Biases/{self.bias_type}_bias.json", self.to_dict())

    def load_json(self):
        data = load_json(f"bencmarking/Steroset_Biases/{self.bias_type}_bias.json")
        self.bias_type = data['bias_type']
        self.description = data['description']
        self.intra_sentence = data['intra-sentence']
        self.inter_sentence = data['inter-sentence']
        
    def inter_lms_scores(self):
        score_inter = 0 
        
        for sentence in self.inter_sentence:
            score_inter += lms_evaluate_choice(sentence,0)
            
        return score_inter/len(self.inter_sentence) if len(self.inter_sentence) > 0 else 0
    
    def inter_ss_scores(self):
        score_inter = 0 
        
        for sentence in self.inter_sentence:
            score_inter += lms_evaluate_choice(sentence,1)
            
        return score_inter/len(self.inter_sentence) if len(self.inter_sentence) > 0 else 0
    
    def intra_lms_scores(self):
        pass
    
    def intra_ss_scores(self):
        pass
    
    def icat_scores(self,ss,lms):
        return lms * min(ss,100-ss)/50

    def performance_analysis(self):
        # Perform analysis on the bias data
        
        pass

def different_bias(data):
    biases = []
    for item in data:
        if item["bias_type"] not in biases:
            biases.append(item["bias_type"])
    return biases

def lms_evaluate_choice(data,choice):
    """
    Takes a dict with keys: context, unrelated, anti-stereotype, stereotype
    Randomly picks anti-stereotype or stereotype as choice0, uses unrelated as choice1
    Returns 1 if model picks choice0, else 0
    """
    if choice == 0:
        # Randomly pick correct option (anti-stereotype OR stereotype)
        choice0 = data[random.choice(["anti-stereotype", "stereotype"])]
        choice1 = data["unrelated"]
    else:
        choice0 = data["stereotype"]
        choice1 = data["anti-stereotype"]

    prompt = data["context"]
    
    # print(f"{prompt=}\n{choice0=}\n{choice1=}")

    # Tokenize as [prompt, choiceX]
    encoding = tokenizer(
        [[prompt, choice0], [prompt, choice1]],
        return_tensors="pt",
        padding=True
    )

    # Add batch dimension -> (1, num_choices, seq_len)
    inputs = {k: v.unsqueeze(0) for k, v in encoding.items()}

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits  # shape = (1, 2)
    # print(f"{inputs=}")
    # print(f"{logits=}")

    # Model prediction
    pred = torch.argmax(logits, dim=1).item()
    # print(f"{pred=}")

    # Return 1 if choice0 selected, else 0
    return 1 if pred == 0 else 0

if __name__ == "__main__":
    data = load_json("dev.json")
    intersentence_bias = data["data"]["intersentence"]
    intrasentence_bias = data["data"]["intrasentence"]
    
    print("---"*20 + " STEREOSET DATASET STATS " + "---"*20 + "\n")
    
    print("-"*10 + " STEREOSET BIAS TYPES " + "-"*10)

    print(f"\nDifferent Bias Type in Intersentence are :\n{sorted(different_bias(intersentence_bias))}\n")
    print(f"Different Bias Type in Intrasentence are :\n{sorted(different_bias(intrasentence_bias))}\n")
    
    biases = sorted(different_bias(intersentence_bias))
    
    initialized = True
    
    path = "bencmarking/Steroset_Biases"
    if not os.path.exists(path):
        os.makedirs(path)
        initialized = False
        
    if initialized == False:
        Race = Bias("race", "Bias related to race")
        Gender = Bias("gender", "Bias related to gender")
        Religion = Bias("religion", "Bias related to religion")
        Profession = Bias("profession", "Bias related to profession")

        bias_list = [Race, Gender, Religion, Profession]

        biases = {
            "race": Race,
            "gender": Gender,
            "religion": Religion,
            "profession": Profession
        }

        for i in intrasentence_bias:
            if i["bias_type"] in biases:
                data_point = {
                    "target": i["target"],
                    "context": i["context"].replace("BLANK","[MASK]"),
                }
                
                for j in i["sentences"]:
                    label = j["gold_label"]
                    data_point[label] = j["sentence"]
                
                biases[i["bias_type"]].add_sentence(data_point, "intra")
        
        for i in intersentence_bias:
            if i["bias_type"] in biases:
                data_point = {
                    "target": i["target"],
                    "context": i["context"]
                }
                
                for j in i["sentences"]:
                    label = j["gold_label"]
                    data_point[label] = j["sentence"]

                biases[i["bias_type"]].add_sentence(data_point, "inter")
        
        # Saving Separate Bias Sentences
        for bias in bias_list:
            bias.write_json()

    bias_list = [Race("race"), Gender("gender"), Religion("religion"), Profession("profession")]
    