import pandas as pd
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1
        
def HitRate(recommendations, actual):
    hits = 0
    for v in list(set(actual)):
        if find_element_in_list(v, recommendations) > 0:
            hits +=1
    return hits / len(recommendations)

def Precision(recommendations, actual):
    hits = len(list(set(actual) & set(recommendations)))
    return hits / len(list(set(recommendations)))

def Recall(recommendations, actual):
    hits = len(list(set(actual) & set(recommendations)))
    return hits / len(list(set(actual)))

def FMeasure(recommendations, actual):
    p   =   Precision(recommendations, actual)
    r   =   Recall(recommendations, actual)
    if( p == 0 or r == 0):
        return 0
    return (2*p*r)/ (p + r)

def Metrics(recommendations, actual):
    prec = Precision(recommendations, actual),
    rec = Recall(recommendations, actual),
    hr = HitRate(recommendations, actual),
    f1 = FMeasure(recommendations, actual),
    return [prec, rec, hr, f1]
    