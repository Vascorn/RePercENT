import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sklearn.linear_model import LogisticRegression



def linearprobe(train_dataset, test_dataset):
    train_zs, train_z1x, train_z2y = getemb(train_dataset)
    test_zs, test_z1x, test_z2y = getemb(test_dataset)
    score_s = linearprobe_acc(train_zs, test_zs, train_dataset, test_dataset)
    score_spe1 = linearprobe_acc(train_z1x, test_z1x, train_dataset, test_dataset)
    score_spe2 = linearprobe_acc(train_z2y, test_z2y, train_dataset, test_dataset)
    return (score_s, score_spe1, score_spe2)

def linearprobe_acc(train_z, test_z, train_dataset, test_dataset):
    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-3])
    score1 = clf.score(test_z, test_dataset[:][-3])
    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-2])
    score2 = clf.score(test_z, test_dataset[:][-2])
    clf = LogisticRegression(max_iter=200).fit(train_z, train_dataset[:][-1])
    score3 = clf.score(test_z, test_dataset[:][-1])
    return (score1, score2, score3)