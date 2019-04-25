import pandas as pd
def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1
        
def Precision(topn, test):
  num_intersect = len([value for value in topn if value in test])
  num_rec       = len(topn)
  return num_intersect / num_rec

def Recall(topn, test):
  num_intersect = len([value for value in topn if value in test])
  num_test       = len(test)
  return num_intersect / num_test

def Hitrate(topn, test):
  num_intersect = len([value for value in list(set(test)) if value in topn])
  num_rec       = len(topn)
  return num_intersect / num_rec

def FMeasure(prec, rec):
  return (2 * prec * rec) / (prec + rec)

    