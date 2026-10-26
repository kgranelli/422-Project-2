import numpy as np
from collections import Counter


def loadTextDataMC(filename, illegalWords={}):
    wfreq = Counter()
    h = open(filename, encoding='latin-1')
    D = []
    for l in h.readlines():
        a = l.split()
        if len(a) > 1:
            y = int(a[0])
            x = {}
            for w in a[1:]:
                if w not in illegalWords:
                    x[w] = 1
            for w in x.keys():
                wfreq[w] += 1
            D.append( (x,y) )
    h.close()

    wid = {}
    widr = []
    maxId = 1
    for w, c in wfreq.items():
        if c >= 20 and c < 0.7 * len(D):
            wid[w] = maxId
            widr.append(w)
            maxId += 1

    N = len(D)

    Xall = np.zeros((N, maxId-1), dtype=float)
    Yall = np.zeros((N,), dtype=int)
    for n in range(len(D)):
        (x, y) = D[n]
        Yall[n] = y
        for w in x.keys():
            if w in wid:
                Xall[n, wid[w] - 1] = 1

    return Xall, Yall, widr


class WineData:
    labels = ['Sauvignon-Blanc', 'Cabernet-Sauvignon', 'Pinot-Noir', 'Pinot-Gris', 'Pinot-Grigio', 'Chardonnay', 'Brut', 'Merlot', 'Shiraz', 'Malbec', 'Zinfandel', 'Cuvee', 'Riesling', 'Chianti', 'Syrah', 'Blend', 'Rhone', 'Viognier', 'Carmenere', 'Moscato']
    
    illegalWords = {'sauvignon': 1, 'blanc': 1, 'cabernet': 1, 'sauvignon': 1, 'pinot': 1, 'noir': 1, 'pinot': 1, 'gris': 1, 'pinot': 1, 'grigio': 1, 'chardonnay': 1, 'brut': 1, 'merlot': 1, 'shiraz': 1, 'malbec': 1, 'zinfandel': 1, 'cuvee': 1, 'riesling': 1, 'chianti': 1, 'syrah': 1, 'blend': 1, 'rhone': 1, 'viognier': 1, 'carmenere': 1, 'moscato': 1}
    
    Xall, Yall, words = loadTextDataMC('data/wines.alldata', illegalWords)
    N, D = Xall.shape
    N0 = int(float(N) * 0.5)
    X = Xall[0:N0, :]
    Y = Yall[0:N0]
    Xte = Xall[N0:, :]
    Yte = Yall[N0:]


class TinyWineData:
    labels = WineData.labels[0:5]
    X      = WineData.X[WineData.Y < 5, :]
    Y      = WineData.Y[WineData.Y < 5]
    Xte    = WineData.Xte[WineData.Yte < 5, :]
    Yte    = WineData.Yte[WineData.Yte < 5]
    words  = WineData.words


class WineDataBinary:
    labels = WineData.labels[0:2]
    X      = WineData.X[WineData.Y < 2, :]
    Y      = 2 * (WineData.Y[WineData.Y < 2] == 0) - 1
    Xte    = WineData.Xte[WineData.Yte < 2, :]
    Yte    = 2 * (WineData.Yte[WineData.Yte < 2] == 0) - 1
    words  = WineData.words



class TwoDDiagonal:
    X = np.array([[ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 0.1,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 0.1, 0.2,
    0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 0.1, 0.2, 0.3,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1. , 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    0.7, 0.8, 0.9, 1. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
    0.8, 0.9, 1. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.9, 1. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1. ],
    [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2,
    0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3,
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4,
    0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6,
    0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
    0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
    0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
    0.9, 0.9, 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. ,
    1. ]]).T - 0.45
    Y = (X[:,0] + 3 * X[:,1] > 0) * 2 - 1.
    Xte = X - 0.05
    Yte = (Xte[:,0] + 3 * Xte[:,1] > 0.4) * 2 - 1.