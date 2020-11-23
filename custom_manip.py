import re
import pandas as pd
import numpy as np

def get_cols(df, terms, ending = False):
    if isinstance(terms, list):
        if not ending:
            last = '^' + terms[-1]
            terms = ['^' + sub + '|' for sub in terms]
            terms[-1] = last
            pattern = ''.join(terms)
    else:
        pattern = '^' + terms

    mask = df.columns.map(lambda x: bool(re.match(pattern, x)))
    return df.columns[mask]

def rebalance(data, output):
    counts = data[output].value_counts()
    n_samples = min(counts)
    ret = pd.DataFrame()

    for val in counts.index:
        df = data.loc[data[output] == val]
        ret = pd.concat([ret, df.sample(n = n_samples)], axis=0)

    return ret

def get_dummies(data, catg, drop_first = True):

    ret = pd.DataFrame()
    ret_names = []

    for cat in catg:

        col = data[cat]
        arr = pd.get_dummies(col, drop_first=drop_first)
        names = np.sort(col.unique())

        if drop_first:
            names = names[1:]

        ret = pd.concat([ret, arr], axis=1)
        ret_names += [str(cat) + '_' + str(n) for n in names]

    ret.columns = ret_names
    data = data.drop(catg, axis=1)

    return pd.concat([data, ret], axis=1)