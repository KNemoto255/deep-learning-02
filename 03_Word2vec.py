"""
Word2Vec
カウントベース手法を使って「単語の意味」を抽出
→分布仮説に基づいて単語の共起行列を求め、その相互情報量・特異値分解後の値を求める
カウントベース手法の代わりに、推論ベース手法を使って計算量を減らす

一般的な言語は、語数が約100万ほどとなる。
カウントベース手法ではコーパスを作ると要素数が1兆、特異値分解を行う際の計算量が1澗(10^36, Sixtillion)を超えてしまい、
べらぼうな計算量が必要となる。代わりに、ニューラルネットワークの利点を生かしてミニバッチ学習を行い、計算量を減らす

出現する単語を予測するには確率分布を算出できるモデルを用いる。確率分布を算出できるモデルの一つがニューラルネットワークである
"""
#単語ベクトルを3つの出力に変換するニューラルネットワークを描く
import numpy as np

c = np.array([1,0,0,0,0,0,0])
W = np.random.randn(7,3)    #ウェイトをランダムで設定
h = np.dot(c, W)            #中間ノード - 全結合層
print(h)

"""
ニューラルネットワークを使って単語を予測するための確率分布を算出するには、
Continuous bag of words（CBOW、連続的な単語の組み合わせ）というモデルを用いる。
CBOWをニューラルネットワークをを使って学習させることで、カウントベース手法と同様に単語の分散表現を獲得できる。
"""
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul

# サンプルのコンテキストデータ
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 重みの初期化
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# レイヤの生成
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 順伝搬
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)
print(s)

"""
単語と単語の間にある単語を正確に予測できるか、という問いをニューラルネットワークで解く。
単語の前・後ろの単語をコンテキスト、正解に相当する単語をターゲットとし、それらをワンホットベクトルとする
"""

import sys
from common.util import preprocess, create_contexts_target, convert_one_hot

text = "When I was 17, I read a quote that went something like: If you live each day as if it was your last, someday you’ll most certainly be right. It made an impression on me, and since then, for the past 33 years, I have looked in the mirror every morning and asked myself: “If today were the last day of my life, would I want to do what I am about to do today?” And whenever the answer has been “No” for too many days in a row, I know I need to change something.Remembering that I’ll be dead soon is the most important tool I’ve ever encountered to help me make the big choices in life. Because almost everything – all external expectations, all pride, all fear of embarrassment or failure – these things just fall away in the face of death, leaving only what is truly important. Remembering that you are going to die is the best way I know to avoid the trap of thinking you have something to lose. You are already naked. There is no reason not to follow your heart."
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

#コンテキスト
print("コンテキスト")
print(contexts)
#ターゲット
print("ターゲット")
print(target)

"""
単語と単語の間にある単語を正確に予測できるか、という問いをニューラルネットワークで解く。
単語の前・後ろの単語をコンテキスト、正解に相当する単語をターゲットとし、それらをワンホットベクトルとする
"""

import sys
from common.util import preprocess, create_contexts_target, convert_one_hot

text = "When I was 17, I read a quote that went something like: If you live each day as if it was your last, someday you’ll most certainly be right. It made an impression on me, and since then, for the past 33 years, I have looked in the mirror every morning and asked myself: “If today were the last day of my life, would I want to do what I am about to do today?” And whenever the answer has been “No” for too many days in a row, I know I need to change something.Remembering that I’ll be dead soon is the most important tool I’ve ever encountered to help me make the big choices in life. Because almost everything – all external expectations, all pride, all fear of embarrassment or failure – these things just fall away in the face of death, leaving only what is truly important. Remembering that you are going to die is the best way I know to avoid the trap of thinking you have something to lose. You are already naked. There is no reason not to follow your heart."
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

#コンテキスト
print("コンテキスト")
print(contexts)
#ターゲット
print("ターゲット")
print(target)

#学習コードの実装
import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot


window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = "When I was 17, I read a quote that went something like: If you live each day as if it was your last, someday you’ll most certainly be right. It made an impression on me, and since then, for the past 33 years, I have looked in the mirror every morning and asked myself: “If today were the last day of my life, would I want to do what I am about to do today?” And whenever the answer has been “No” for too many days in a row, I know I need to change something.Remembering that I’ll be dead soon is the most important tool I’ve ever encountered to help me make the big choices in life. Because almost everything – all external expectations, all pride, all fear of embarrassment or failure – these things just fall away in the face of death, leaving only what is truly important. Remembering that you are going to die is the best way I know to avoid the trap of thinking you have something to lose. You are already naked. There is no reason not to follow your heart."
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs

print("単語：ベクトル")
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

import matplotlib.pyplot as plt
for word, word_id in word_to_id.items():
    plt.annotate(word, (word_vecs[word_id, 0], word_vecs[word_id,1]))

plt.scatter(word_vecs[:,0], word_vecs[:,1], alpha=0.5)
plt.figure(figsize=(20,20))
plt.show()

"""
CBOWモデルでニューラルネットワークを学習するには、損失関数に交差エントロピー誤差ではなく負の尤度対数(Negative log likehood)を
用いる。
"""






