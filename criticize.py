"""
Turn harmless labels into reasons you suck
"""
# comment
import re

def interpret_label(label:str) -> str:
    """Translate a single label into an interpretation"""
    lbl = label.lower()
    # set the default interpretation
    interpretation = "wasting time"

    if lbl == "friends":
        interpretation = "shenanigans and gallivanting"
    elif lbl == "food":
        interpretation = "wasting money"
    elif lbl in ["dog", "cat", "pet"]:
        interpretation = "dirty animals"
    elif lbl in ["floor", "wall", "home", "house", "indoors"]:
        interpretation = "visiting other people's houses and showing no manners"
    return interpretation


def interpret_labels(labels:list) -> list[str]:
    """Batch interpret labels"""
    interpretations = set()
    for lbl in labels:
        interpretations.add(interpret_label(lbl))
    return list(interpretations)


if __name__ == "__main__":
    # just checking
    labels = ["happiness", "joy", "food"]
    interps = interpret_labels(labels)
    print(interps)