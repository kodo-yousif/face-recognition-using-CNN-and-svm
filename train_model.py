import numpy as np
from prettytable import PrettyTable
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

log_file = "metrics_log.txt"

def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    return eer

def train_model(pipeline, X, y, kfold):
    table = PrettyTable()
    table.field_names = ["Fold", "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "EER"]

    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': [], 'eer': []}

    for fold, (train, test) in enumerate(kfold.split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        y_test_binarized = label_binarize(y_test, classes=np.unique(y))
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test_binarized, y_prob, multi_class="ovr", average="macro")
        eer = calculate_eer(y_test_binarized.ravel(), y_prob.ravel())

        table.add_row([fold+1, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{auc:.4f}", f"{eer:.4f}"])

        scores['accuracy'].append(acc)
        scores['precision'].append(prec)
        scores['recall'].append(rec)
        scores['f1'].append(f1)
        scores['auc'].append(auc)
        scores['eer'].append(eer)

    means = [np.mean(vals) for vals in scores.values()]
    
    table.add_row(["Mean", f"{means[0]:.4f}", f"{means[1]:.4f}", f"{means[2]:.4f}", f"{means[3]:.4f}", f"{means[4]:.4f}", f"{means[5]:.4f}"])

    print(table)

    with open(log_file, "a") as f:
        f.write(table.get_string()) 
        f.write("\n") 