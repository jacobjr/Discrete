import numpy as np


f = open("../Mediamill_data.txt")
f.readline()
line = f.readline()
labels = []
features = []

while not line == "":
    if not line[0] == " ":
        feats = line.split(" ")[1:]
        fs = []
        for feat in feats:
            fs += [float(feat.split(":")[1])]
        features += [fs]
        labels += [line.split(" ")[0].split(",")]
    line = f.readline()
labs = np.zeros((len(features), 101))
for i in range(0, labs.shape[0]):
    labs[i,[int(x) for x in labels[i]]] = 1
features = np.array(features)
labels = labs

f.close()
np.savetxt("features.data", features, fmt="%10.5f")
np.savetxt("labels.data", labels, fmt="%d")
