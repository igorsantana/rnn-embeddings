def Precision(topn, test):
  num_intersect = len(set.intersection(set(topn), set(test)))
  num_rec       = len(topn)
  return num_intersect / num_rec

def Recall(topn, test):
  num_intersect   = len(set.intersection(set(topn), set(test)))
  num_test        = len(list(set(test)))
  return num_intersect / num_test

def Hitrate(topn, test):
  num_intersect = len([value for value in list(set(test)) if value in topn])
  num_rec       = len(topn)
  return num_intersect / num_rec

def FMeasure(prec, rec):
  return (2 * ((prec * rec) / (prec + rec))) if (prec + rec) > 0 else 0

    