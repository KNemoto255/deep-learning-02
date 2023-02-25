"""
ニューラルネットワークの基礎となる、線形代数・微積分、
順伝播と誤差逆伝播について学ぶ

ニューラルネットワークでは、入力・出力されるデータは主にベクトルや行列・テンソルといった
多次元の数値データとして扱われる。これらのデータが画像や文章といったデータ型・ファイルの形
となることで意味のある情報となる。原理的にはこれらの情報・アルゴリズムは機械語レベルまで分解され、0か1かのビット情報として
CPUやGPU・記憶媒体の値として処理される。

Pythonでは、ベクトルや行列を扱うのにNumpyを用いる
"""

"""
Numpyを用いた四則演算・行列演算
"""

import numpy as np
X = np.array([[1,2,3],[4,5,6]])
W = np.array([[0,1,2],[3,4,5]])

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
#計算に必要なパラメータがどうなっているかを確認する
x = np.random.randn(10,2)
W1 = np.random.randn(2,4)
b1 = np.random.randn(4)

print(x)
print(W1)
print(b1)

#全結合層 - 活性化関数を掛けず、線的な変換を行う
h = np.dot(x,W1) + b1
print(h)

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

print(x)
print(W1)
print(b1)
print(W2)
print(b2)

#全結合層 - 活性化関数を掛けた値を最後のノードに送る
h = np.dot(x,W1) + b1
a = sigmoid(h)
s = np.dot(a,W2) + b2

"""
隠れ層が2層の順伝播型ニューラルネットワークをオブジェクト指向で実装
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

#TwoLayerNetを使って推論を行う
x = np.random.randn(10,2)
model = TwoLayerNet(2,4,3)
s = model.predict(x)

"""
ニューラルネットワークの学習 - 誤差逆伝播のアルゴリズムを実装
ニューラルネットワークの予測がどれだけ謝っているかを表すのには、損失(loss)という指標を用いる。
損失を求める関数には、多クラス分類を行う場合交差エントロピー誤差(Cross Entropy Error)をつかう
交差エントロピー誤差を使って誤差逆伝播をする機械学習の手法は、ニューラルネットワークの出力する値を
確率分布を推定した値と見做し、実際の確率分布に合わせて誤差を修正するというベイズ統計(確率のベイズ的解釈に基づく統計学)に
基づいたモデルとなっている

ニューラルネットワークの学習は、損失＝交差エントロピー誤差を減らすことを目的とする。そうすることで、データを基に正確な予測ができる
モデルが完成する。モデルは100%正しくすることが目的ではなく、なるべく有用な近似を発見することを目指す

交差エントロピー誤差をもとに誤差をノードへ逆伝播するには、誤差を微分した値をパラメータから引く
誤差を微分した値を末端まで伝播することで、事後確率に合わせたパラメータの値に更新される

誤差逆伝播：https://qiita.com/kenta1984/items/59a9ef1788e6934fd962
ベイズ統計学：https://ja.wikipedia.org/wiki/%E3%83%99%E3%82%A4%E3%82%BA%E7%B5%B1%E8%A8%88%E5%AD%A6
"""
import sys
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print("x", x.shape)
print("t", t.shape)

# データ点のプロット
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()

# coding: utf-8
import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


