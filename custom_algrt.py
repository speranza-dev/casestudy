import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

def classify(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:, 1]])

def nested_cross_validation(X, y, numl, model, space, ts = [0.25,0.5,0.75], scaler = None):
    
    cols = X.columns
    X = np.array(X)
    y = np.array(y)
                        
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)

    auc_scores, acc_scores, f1_scores = list(), list(), list()
    for train_ix, test_ix in cv_outer.split(X):

        X_train, X_vald = X[train_ix], X[test_ix]
        y_train, y_vald = y[train_ix], y[test_ix]

        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        
        if(scaler is not None):
            X_scl = scaler.fit(X_train[numl])
            X_train[numl] = X_scl.transform(X_train[numl])
        
        search = GridSearchCV(model, space, scoring='f1_micro', cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        #yhat = result.predict(pd.DataFrame(X_vald, columns=cols))

        if(scaler is not None):
            X_scl = scaler.fit(X_vald[numl])
            X_vald[numl] = X_scl.transform(X_vald[numl])
        
        probs = result.predict_proba(X_vald)
        max_score, best_t = 0, 0
        
        for t in ts:
            y_pred = classify(probs, t)
            f1 = f1_score(y_vald, y_pred)
            if(f1 > max_score):
                max_score = f1
                best_t = t
        
        y_pred = classify(probs, best_t)
        acc_score = accuracy_score(y_vald, y_pred)
        roc_auc = roc_auc_score(y_vald, y_pred)
        
        #y_hat = result.predict(X_vald)
        #auc_score = roc_auc_score(y_vald, y_hat)

        auc_scores.append(roc_auc)
        acc_scores.append(acc_score)
        f1_scores.append(max_score)
        
        print(">> validation f1 = {}, train f1 = {}, threshold = {}, clf = {}".format(
            round(max_score,2), round(result.best_score_,2), best_t, result.best_params_))

    print('Mean validation results:')
    print('F1: %.3f (%.3f)' % (np.mean(f1_scores), np.std(f1_scores)))
    print('AUC: %.3f (%.3f)' % (np.mean(auc_scores), np.std(auc_scores)))
    print('Accuracy: %.3f (%.3f)' % (np.mean(acc_scores), np.std(acc_scores)))
    
    
def print_clf_metrics(labels, scores):
    metrics = precision_recall_fscore_support(labels, scores)
    conf = confusion_matrix(labels, scores)
    print('                 Confusion Matrix')
    print('                 Pred. Negative    Pred. Positive')
    print('True Negative   %6d' % conf[0, 0] + '             %5d' % conf[0, 1])
    print('True Positive   %6d' % conf[1, 0] + '             %5d' % conf[1, 1])
    print('')
    print('Accuracy   %0.2f' % accuracy_score(labels, scores))
    print('ROC-AUC    %0.2f' % roc_auc_score(labels, scores))
    print(' ')
    print('             Positive      Negative')
    print('Num. cases %6d' % metrics[3][1] + '      %6d' % metrics[3][0])
    print('Precision  %6.2f' % metrics[0][1] + '        %6.2f' % metrics[0][0])
    print('Recall     %6.2f' % metrics[1][1] + '        %6.2f' % metrics[1][0])
    print('F1         %6.2f' % metrics[2][1] + '        %6.2f' % metrics[2][0])
    
def xgb_diagnostics(X_train, X_test, y_train, y_test, params, n_repeats = 30, plot_title = ''):
     
    aucs_test, aucs_train = [],[]
    for i in range(n_repeats):
        
        metric = 'aucpr'
        model = XGBClassifier(**params, random_state=i, n_jobs=-1)
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_metric=[metric], eval_set=eval_set, verbose=False)

        results = model.evals_result()
        test = results['validation_1'][metric]
        train = results['validation_0'][metric]
        aucs_test += [test[-1]] 
        aucs_train += [train[-1]]

    return aucs_test
    
def vary_xgb_params(X_train, X_test, y_train, y_test, params, to_vary, values):
    
    results = pd.DataFrame()
    
    for val in values:
        params[to_vary] = val
        aucs = xgb_diagnostics(X_train, X_test, y_train, y_test, params, n_repeats=30, plot_title = to_vary + ' = ' + str(val))
        print("[Experiment {}={}] Average Test PR-AUC: {}".format(to_vary,val,np.mean(aucs)))
        results[str(val)] = aucs
        
    results.boxplot()