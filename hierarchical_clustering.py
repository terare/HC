#This is a code for hierarchical clustering of stored features.
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import pickle
from imagenet1k_dict import imagenet1k_dict
import argparse
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', choices=['clip','vit','wrn'], default='clip')
parser.add_argument('-sl','--semantic_layer', type=int, default=0)
parser.add_argument('-cm','--print_confusion_matrix', default=True)
args = parser.parse_args()


if __name__ == '__main__':

    if args.model == 'clip':
        f = open("features/clip.pickle","rb")
    elif args.model == 'vit':
        f = open("features/vit.pickle","rb")
    elif args.model == 'wrn':
        f = open("features/wrn.pickle","rb")
    else:
        print('Warning: The specified model does not exist.')
    features = pickle.load(f)

    num_class = 1000
    # 以下はimagenetのラベル特有の処理．複数の呼び名があるものはその数だけ
    # ラベルが付与されているが，図に乗せるときには見やすさのため最初の
    # 一つのみ表示するようにしている．ex)'tench, Tinca tinca'->'tench'
    class_names = []
    for i in range(num_class):
        r = imagenet1k_dict[i]['label']
        if ',' in r:
            idx = r.find(',')
            r = r[:idx]
        text = '{}:{}'.format(i,r)
        class_names.append(text)

    plt.clf()
    plt.figure(figsize=(300,50))
    single_linkage = linkage(features, method='ward')
    dn = dendrogram(single_linkage.tolist(),color_threshold=80,labels=class_names,leaf_font_size=10) #clipの場合thresholdの値は80にしておく
    
    # Color the labels on the dendrogram
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    gt = [0]*num_class # others(0)かartifact(1)かanimal(2)かのGTを格納する．
    for lbl in xlbls:
        # labelのnumberを取得．ex)'558:flute'->558
        r = lbl.get_text()
        if ':' in r:
            idx = r.find(':')
            r = int(r[:idx])
        # 1番上の階層でgroupを作る
        if args.semantic_layer == 0:
            if imagenet1k_dict[r]['multi-label'][0] == 'animal':
                lbl.set_color('r')
                gt[r] = 2
            elif imagenet1k_dict[r]['multi-label'][0] == 'artifact':
                lbl.set_color('g')
                gt[r] = 1
            else:
                lbl.set_color('b')
        if r == 241:    # 6/9 EntleBucherをanimalに追加(このクラスは本来animalに属するため)
            lbl.set_color('r')
    plt.title('{}'.format(args.model),fontsize=30)
    plt.show()
    if not os.path.isdir('results'):
        os.makedirs('results')
    plt.savefig("results/dengra_{}.png".format(args.model))

    # Output the confusion matrix
    if args.print_confusion_matrix == True:
        result = fcluster(single_linkage, t=2, criterion="maxclust").tolist()
        print(confusion_matrix(gt,result))

    print("Done!")