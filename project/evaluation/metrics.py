from project.evaluation.ranking_metrics import mean_average_precision, ndcg_at, precision_at

def __Prec(topn, test):
  num_intersect = len(set.intersection(set(topn), set(test)))
  num_rec       = len(topn)
  return num_intersect / num_rec

def __Rec(topn, test):
  num_intersect   = len(set.intersection(set(topn), set(test)))
  num_test        = len(list(set(test)))
  return num_intersect / num_test

def Hitrate(topn, test):
  num_intersect = len([value for value in list(set(test)) if value in topn])
  num_rec       = len(topn)
  return num_intersect / num_rec

def __F1(prec, rec):
  return (2 * ((prec * rec) / (prec + rec))) if (prec + rec) > 0 else 0

  
def get_metrics(topn, test):
  prec    = __Prec(topn, test)
  rec     = __Rec(topn, test)
  f       = __F1(prec, rec)
  MAP     = mean_average_precision([test], [topn], assume_unique=False)
  ndcg_5  = ndcg_at([test], [topn], k=5, assume_unique=False)
  p_5     = precision_at([test], [topn], k=5, assume_unique=False)
  
  return [prec, rec, f, MAP, ndcg_5, p_5]