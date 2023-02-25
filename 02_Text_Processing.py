"""
自然言語と単語の分散表現

コンピューターが機械言語ではない、人間の言語を理解するにはどのようなモデル・アルゴリズムを考えればいいか
という本質的な問題を考える。まずテキストをPythonで扱えるようにする他、テキストを単語に分割・単語を単語IDに変換する処理を覚える

自然言語の難しさは、プログラミング言語と違い曖昧さ・コンテキスト・同じ文章での意味の違い等々の「柔らかさ」を
データ化することが難しいことにある。この柔らかさが、人が適当かつ迅速な思考・行動をすることを可能にしている。

言語はまず文字によって構成され、言葉の意味は単語によって構成される。自然言語をコンピューターで扱いには、
「単語の意味」を何らかの方法でデータ化する必要がある。
これまでに単語の意味を表現するには、シソーラス・カウントべース・推論べ―ス(word2vec)という手法が考案されている。
今回はカウントベース、推論ベースの手法を実装する
"""

"""
テキストデータを扱うには、まずテキストの集まりであるコーパスが必要。
コーパスにはテキストの本質である「意味」や「知識」といった、単なる文字の集まりではないデータが含まれている、と見做す
ことができる

まず、テキストをデータに変換する方法を考える
"""
#テキストを全て小文字に＋ピリオドを正規化する
text = "You say goodbye and I say hello."
text = text.lower()
text = text.replace(",", " ,")

print("テキスト")
print(text)

#空白で区切る
words = text.split(" ")
print("空白区切り")
print(words)

#単語をインデックスを対応させる
word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print("単語＋単語ID")
print(word_to_id)
print(id_to_word)

#文章を単語から単語のID=コーパスに変換する
import numpy as np
corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)

print("コーパス")
print(corpus)

"""
テキストを処理してコーパスのリスト・{単語：単語ID}の辞書データに変換する関数
"""
import numpy as np
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

print("テキスト")
print(text)
print("コーパス")
print(corpus)

"""
単語の分散表現
コンピューターでは、色をRGB - 3成分のベクトルのデータとして処理する。
色をデジタル化できるのと同様、「単語の意味」をベクトルデータ化（単語の分散表現）にできないか考える。

言語に関する仮説
生成文法 - 人間は言語の初期状態である普遍文法（Universal grammar)を生得的に備えている
文章は句構造文法となっていて、音と意味の結びつき・二語以上の語からなる統語構造などから成る

言語的相対論、サピア＝ウォーフの仮説 - 言語はその話者の世界観の形成に関与する。使用できる言語によってその個人の思考が影響を受ける

シニフィアンとシニフィエ:シーニュ - テキストは語のもつ感覚的側面：シニフィアンとシニフィアンによって意味されるイメージ、意味：シニフィエに分かれている。
シニフィアンとシニフィエの対がシーニュ：記号であり、言語や記号表現を成立させるための最小単位となる

分布仮説 - 単語の意味は、その単語の周囲の単語によって形成される。テキストが厳密な句構造文法やの思考との相関関係・シーニュである必然性はない
単語自体には意味がなく、その単語の組み合わせとコンテキストによって単語の意味が形成される
"""

import sys
import numpy as np
from common.util import preprocess

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

print("テキスト")
print(text)
print("コーパス")
print(corpus)

"""
単語の周囲にどれだけ単語が出現するかをカウントすると、その単語のデータをベクトルかできる。
単語のベクトル・単語の出現数を合わせた行列を、共起行列という。
単語がベクトルできたら、そのコサイン類似度を比較する。
"""
#コーパスから共起行列を作成
def create_co_matrix(corpus, vocab_size, window_size=1):
    '''共起行列の作成
    :param corpus: コーパス（単語IDのリスト）
    :param vocab_size:語彙数
    :param window_size:ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return: 共起行列
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

#コサイン類似度を比較
def cos_similarity(x, y, eps=1e-8):
    '''コサイン類似度の算出
    :param x: ベクトル
    :param y: ベクトル
    :param eps: ”0割り”防止のための微小値
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

#コサイン類似度を求める
import sys
text = "When I was 17, I read a quote that went something like: If you live each day as if it was your last, someday you’ll most certainly be right. It made an impression on me, and since then, for the past 33 years, I have looked in the mirror every morning and asked myself: “If today were the last day of my life, would I want to do what I am about to do today?” And whenever the answer has been “No” for too many days in a row, I know I need to change something.Remembering that I’ll be dead soon is the most important tool I’ve ever encountered to help me make the big choices in life. Because almost everything – all external expectations, all pride, all fear of embarrassment or failure – these things just fall away in the face of death, leaving only what is truly important. Remembering that you are going to die is the best way I know to avoid the trap of thinking you have something to lose. You are already naked. There is no reason not to follow your heart."

corpus, word_to_id, id_to_word = preprocess(text)

print("テキスト")
print(text)
print("コーパス")
print(corpus)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

#youとiの単語ベクトルを求める
c0 = C[word_to_id["you"]]
c1 = C[word_to_id["i"]]

#コサイン類似度を算出
print(cos_similarity(c0, c1))

"""
類似する単語をランキング順にする
クエリを取り出す → コサイン類似度を算出する → コサイン類似度の高い順に出力する
"""
def most_similar(query, word_to_id, id_to_word, word_matrix, top=10):
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

most_similar("you", word_to_id, id_to_word, C , top = 10)

"""
単語を全てベクトルにするという、カウントベース手法を若干改善する
"a","the"の様な多数出現する単語に対しマイナス補正をつける、相互情報量(PMI)という指標を使う
PMI(x,y) = log2 P(x,y) /P(x)*P(y) 
PMIは対数なので、共起回数が0となるとマイナス無限大となってしまう。
PMIがマイナス以下となる時は0として扱う（正の相互情報量、PPMI）
"""

def ppmi(C, verbose=False, eps = 1e-8):
    #PPMIに変換するための空の行列を作る
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    #共起回数をPPMIに変換する
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M

import sys
import numpy as np

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

#有効桁を3桁にする
np.set_printoptions(precision=3)
print("共起行列")
print(C)
print("-"*30)
print("相互情報量")
print(W)

"""
単語の相互情報量を求めるとほぼすべてのベクトルが0となってしまうので、より有意義な情報に変換するために
次元削減のアルゴリズムを使う。ここでは、特異値分解(Singular Value Decomposition)を用いる。
SVDを行うには、Numpyのlinalgモジュール・svdメソッドを用いる。
"""
U,S,V = np.linalg.svd(W)
np.set_printoptions(precision=3)
print("共起行列")
print(C)
print("相互情報量")
print(W)
print("特異値分解後の値")
print(U)

"""
特異値分解後の各単語の先頭2行を抽出し、グラフにプロットする
"""
import matplotlib.pyplot as plt
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id,1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()







