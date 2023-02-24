"""
ニューラルネットワークの基礎となる、線形代数・微積分、
順伝播と誤差逆伝播について学ぶ

ニューラルネットワークでは、入力・出力されるデータは主にベクトルや行列・テンソルといった
多次元の数値データとして扱われる。これらのデータが画像や文章といったデータ型・ファイルの形
となることで意味のある情報となる。
原理的にはこれらの情報・アルゴリズムは機械語レベルまで分解され、0か1かのビット情報として
CPUやGPU・記憶媒体の値として処理される。

Pythonでは、ベクトルや行列を扱うのにNumpyを用いる
"""

"""
Numpyを用いた四則演算・行列演算
"""

import numpy as np
X = np.array([1,2,3],[4,5,6])
W = np.array([0,1,2],[3,4,5])

#行列のデータ型・形状・固有ベクトルの数
X.__class__
X.shape
X.ndim

#行列の和
X+W
#行列のスカラー積
X*W
#行列の内積
#スカラー積と違い、行ベクトル・列ベクトルの要素ごとの積と和を求める
#ネットワークにおける推論・学習は、内積にウェイトとバイアスを掛ける処理に相当する

np.dot(X, W)

"""
ニューラルネットワークの推論と学習
ニューロン同士をつなぐ信号には入力値・ウェイトに活性化関数を掛けた値、前層のニューロンには
影響を受けないバイアスを足し合わせた値を用いる
入力層・隠れ層・出力層の3層のニューラルネットワークをシュミレーションするには、
3つの行列を使う必要がある
"""
import numpy as np

#x - 2要素のデータが10個のミニバッチとなっている
x = np.random.randn(10,2)
W1 = np.random.randn(2,4)
b1 = np.random.randn(4)

x
W1
b1

#全結合層 - 活性化関数を掛けず、線的な変換を行う
h = np.dot(x,W1) + b1
h

"""
隠れ層が2層のニューラルネットワークをシュミレーション
2要素のデータを3クラスに分類する
全結合層でシグモイド関数を活性化関数に用いる
"""
import numpy as np

#シグモイド関数 - 非線形的な出力とすることで、より性格な分類ができる
#シグモイド関数を使うことで、線形モデルによる値が入力である場合に0から1の間の値として結果を出力するような回帰式を推定することができる。
#これはロジスティック回帰分析を応用したモデルである
#https://bellcurve.jp/statistics/course/26934.html

def sigmoid(x):
    return 1 / (1 * np.exp(-x))

#x - 2要素のデータが10個のミニバッチとなっている
#最終的な出力値は３要素のデータとなる。
x = np.random.randn(10,2)
W1 = np.random.randn(2,4) #(要素数、出力するデータ数)
b1 = np.random.randn(4)
W2 = np.random.randn(4,3)
b2 = np.random.randn(3)

x
W1
b1
W2
b2

#全結合層 - 活性化関数を掛けた値を最後のノードに送る
h = np.dot(x,W1) + b1
a = sigmoid(h)
s = np.dot(a,W2) + b2

"""
隠れ層が2層のニューラルネットワークをオブジェクト指向で実装
全結合を行う層をAffine、シグモイド関数を使って活性化関数を掛ける層をSigmoidというクラスとして実装する。
また、レイヤーのクラスを使ってニューラルネットワーク全体の構造をオブジェクト指向で実装
"""
import numpy as np
#シグモイド関数、活性化関数
class Sigmoid:
    #コンストラクター
    def __init__(self):
        self.params = []
    #順伝播
    def forward(self, x):
        out = return 1 / (1 + np.exp(-x))
        return out

#全結合層
class Affine:
    #コンストラクター
    def __init__(self, W, b):
        self.params = [W, b]
    #順伝播
    def forward(self, x):
        W, b = self.params
        out = np.dot(x,W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = np.random.randn(I,H) #(要素数、出力するデータ数)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        #各層の内容
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        #各層で計算された値 - パラメータを保管する
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        #予測、プレディクト - 各層で順伝播を行い、出力層の値を計算する
        def predict(self, x):
            for layer in self.layers:
                x = layer.forward(x)
            return x
