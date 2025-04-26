"""
Turn harmless labels into reasons you suck
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
# def interpret_label(label:str) -> str:
#     """Translate a single label into an interpretation"""
#     lbl = label.lower()
#     # set the default interpretation
#     interpretation = "wasting time"

#     if lbl == "friends":
#         interpretation = "shenanigans and gallivanting"
#     elif lbl == "food":
#         interpretation = "wasting money"
#     elif lbl in ["dog", "cat", "pet"]:
#         interpretation = "dirty animals"
#     elif lbl in ["floor", "wall", "home", "house", "indoors"]:
#         interpretation = "visiting other people's houses and showing no manners"
#     return interpretation


# def interpret_labels(labels:list) -> list[str]:
#     """Batch interpret labels"""
#     interpretations = set()
#     for lbl in labels:
#         interpretations.add(interpret_label(lbl))
#     return list(interpretations)

def interpret_labels(regular_labels:list) -> list:
    judgy_labels = ["dishonour on your cow","wasting_money","shenanigans and gallivanting","not calling ","wasting_time", "not studying", "reading picture books", "too skinny", "not skinny enough", "visiting other people's houses and showing no manners","is that ur boyfriend/girlfriend?","Why are you out so late?", "you have to make more money", "you have to eat more healthy", ]
    relevant_judgy_label = []
    for judgylabel in judgy_labels:
        jinputs = tokenizer(judgylabel, return_tensors="pt")
        # jlabels = torch.tensor([1]).unsqueeze(0)  # Add batch dimension
        joutputs = model(**jinputs, output_hidden_states=True)
        # joutputs = model(**inputs)
        last_hidden_states = joutputs.hidden_states[-1]  # Get the last hidden states
        # Get the mean of the last hidden states for the label
        label_embedding = last_hidden_states.mean(dim=1).squeeze().detach().numpy()
        
        # Compare with regular labels
        for regular_label in regular_labels:  
            rinputs = tokenizer(regular_label, return_tensors="pt")
            routputs = model(**rinputs, output_hidden_states=True)
            rlast_hidden_states = routputs.hidden_states[-1]  # Get the last hidden states
            # Get the mean of the last hidden states for the regular label
            regular_label_embedding = rlast_hidden_states.mean(dim=1).squeeze().detach().numpy()
            
            # Calculate cosine similarity
            cosine_similarity = np.dot(label_embedding, regular_label_embedding) / (np.linalg.norm(label_embedding) * np.linalg.norm(regular_label_embedding))
            
            if cosine_similarity > 0.5:
                relevant_judgy_label.append(judgylabel)
        final_list = list(set(relevant_judgy_label))
    return final_list


# if __name__ == "__main__":
#     # just checking
#     labels = ["happiness", "joy", "food"]
#     interps = interpret_labels(labels)
#     print(interps)