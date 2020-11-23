import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cap_outliers(ser, top, bottom):
    low, high = ser.quantile(q = [bottom, top])
    ser[ser < low] = low
    ser[ser > high] = high
    return ser

def plot_hists(data, cols, top=.9, bottom=.1, cap_outs = True, logs=False):
    for col in cols:
        plt.figure()
        if cap_outs:
            cap_outliers(data[col], top, bottom).hist()
        else:
            if(logs):
                pd.DataFrame(np.log(data[col])).hist()
            else:
                data[col].hist()
        plt.title(col)
        plt.show()
        
def plot_bars(data, cols, output):
    for col in cols:
        obj = data.groupby([output, col]).size()
        x1 = np.arange(len(obj[0]))
        x2 = np.arange(len(obj[1]))
        fig, ax = plt.subplots(figsize=(6, 3))
        bar_width = 0.4
        b1 = ax.bar(x1, np.array(obj[0]), label='No', width=bar_width)
        b2 = ax.bar(bar_width + x2, np.array(obj[1]), label='Yes', width=bar_width)
        ax.legend()
        ax.set_xlabel(str(col), labelpad=15)
        ax.set_ylabel('Count', labelpad=15)
        ax.set_title('Counts for ' + str(col), pad=15)
        if len(x1) > len(x2):
            ax.set_xticks(x1 + bar_width / 2)
            ax.set_xticklabels(np.array(obj[0].keys()))
        else:
            ax.set_xticks(x2 + bar_width / 2)
            ax.set_xticklabels(np.array(obj[1].keys()))
        plt.show()
        
        
def heatmap(data, fs = (8,8)):
    plt.figure(figsize=fs)
    plt.imshow(data.corr(), cmap=plt.cm.Blues, interpolation='nearest', aspect='auto')
    plt.colorbar()
    tick_marks = [i for i in range(len(data.columns))]
    plt.xticks(tick_marks, data.columns, rotation='vertical')
    plt.yticks(tick_marks, data.columns)
    plt.show()
    

def plot_box(df, cols, output, suppress_zeros = False):
    for col in cols:
        plt.figure()
        sns.set_style("whitegrid")
        y = df.loc[:,output]
        x = df.loc[:,col]
        if(suppress_zeros):
            mask = x.map(lambda x : True if x > 0 else False)
            y = y[mask]
            x = x[mask]
        sns.boxplot(y, x, data=df)
        plt.xlabel(output)
        plt.ylabel(col)
        plt.show()
        
def plot_percentiles(data, pair, assignment, labs):
    col_dic = {labs[0]:'blue',labs[1]:'green',labs[2]:'orange',labs[3]:'red'}
    colors = [col_dic[x] for x in assignment]
    plt.figure(figsize=(6,3))
    plt.scatter(data[pair[0]], data[pair[1]], color = colors)
    plt.xlabel(pair[0])
    plt.ylabel(pair[1])
    plt.title('')
    plt.show()