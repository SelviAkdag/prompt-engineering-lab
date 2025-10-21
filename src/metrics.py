def precision_recall_f1(y_true, y_pred, positive_label='positive'):
    tp = sum(1 for t,p in zip(y_true, y_pred) if p==positive_label and t==positive_label)
    fp = sum(1 for t,p in zip(y_true, y_pred) if p==positive_label and t!=positive_label)
    fn = sum(1 for t,p in zip(y_true, y_pred) if p!=positive_label and t==positive_label)
    tn = sum(1 for t,p in zip(y_true, y_pred) if p!=positive_label and t!=positive_label)
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    acc = sum(1 for t,p in zip(y_true, y_pred) if t==p)/len(y_true) if y_true else 0.0
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'tp':tp,'fp':fp,'tn':tn,'fn':fn}
