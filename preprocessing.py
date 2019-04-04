from pathlib import Path
import MeCab
import unicodedata
import re
import neologdn
import urllib.request
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)


data_dir_path = Path('.').joinpath('data')
image_dir_path = Path('.').joinpath('image')

if not data_dir_path.exists():
    data_dir_path.mkdir(parents=True)
if not image_dir_path.exists():
    image_dir_path.mkdir(parents=True)


tagger = MeCab.Tagger("-Ochasen")
tagger.parse('')

#  「学問のすすめ」のデータを読み込む
with open(data_dir_path.joinpath('gakumonno_susume.txt'), 'r', encoding='utf-8') as file:
    lines = file.readlines()

#  ストップワードリストを取得する関数
def get_stopwod_lsit(file_name='stopword_list.txt', dir_path='.'):

    file_path = Path(dir_path).joinpath(file_name)
    if not file_path.exists():
        url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
        urllib.request.urlretrieve(url, file_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        stopword__list = [word.replace('\n', '') for word in file.readlines()]

    return stopword__list


def sentence_normalize(sentence):

    sentence = sentence.replace('\n', '')
    sentence = neologdn.normalize(sentence)
    sentence = unicodedata.normalize("NFKC", sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'[!-/:-@[-`{-~]', '', sentence)
    sentence = re.sub(r'\d+\.*\d*', '0', sentence)

    return sentence



def sentence_split(sentence, stopword_list=()):

    sentence = sentence_normalize(sentence)
    node = tagger.parseToNode(sentence)

    words = []
    while node:

        word = node.surface
        features = node.feature.split(",")

        if features[0] == "名詞" and word not in stopword_list:
            if not re.match(r'^[ぁ-んァ-ン]$', word):
                words.append(word)
        node = node.next

    # 文書中の重複はまとめてしまう
    return list(set(words))


stopword_list = get_stopwod_lsit()

splited_setences = []
for text in lines:
    texts = text.split('。')
    for sentence in texts:
        word_lists = sentence_split(sentence, stopword_list=stopword_list)
        if len(word_lists) > 0:
            splited_setences.append(word_lists)


splited_setences_1d = []
for sentence in splited_setences:
    splited_setences_1d.extend(sentence)

#  単語の出現回数を求める
word_count = collections.Counter(splited_setences_1d)

#  単語の出現回数上位30までのグラフを作成
word_count_most = word_count.most_common(30)
words, values = [], []
for word, value in word_count_most:
    words.append(word)
    values.append(value)

fig = plt.figure(figsize=(10, 6))
plt.barh(words, values)
fig.tight_layout()
plt.savefig(image_dir_path.joinpath('word_count_preprocessed.png'),
            bbox_inches="tight")



print(word_count.most_common(30))
n_word = len(word_count)
print('文章数：{}, 単語数：{}'.format(len(splited_setences), n_word))

#  出現回数ごとの単語数を集計
df = pd.DataFrame(word_count.most_common(n_word), columns=['word', 'count'])
df_word_count = df.groupby('count').count()
df_word_count.reset_index(inplace=True)
df_word_count.columns = ['count', 'n_word']
df_word_count['cumsum'] = df_word_count['n_word'].cumsum()
df_word_count['cumsum_rate'] = df_word_count['cumsum'] / df_word_count['n_word'].sum()

print(df_word_count.head())

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# それぞれのaxesオブジェクトのlines属性にLine2Dオブジェクトを追加
ax1.bar(df_word_count['count'], df_word_count['n_word'], color='blue', label="単語数")
ax2.plot(df_word_count['count'], df_word_count['cumsum_rate'], color='red', label="累積割合")

# 凡例
# グラフの本体設定時に、ラベルを手動で設定する必要があるのは、barplotのみ。plotは自動で設定される＞
handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
# 凡例をまとめて出力する
ax1.legend(handler1 + handler2, label1 + label2, loc=2, borderaxespad=0.)
plt.savefig(image_dir_path.joinpath('word_count_cumsum.png'), bbox_inches="tight")

#  単語数を制限
n_word = int(n_word * (1 - df_word_count['cumsum_rate'][2]))

#  出現回数が多い順に番号をふる
word_dicts = dict([[key[0], i] for key, i in zip(word_count.most_common(n_word), range(n_word))])

#  各文章の単語を出現回数ランキング番号に変換する
word_numbers = [[word_dicts[word] for word in words if word_dicts.get(word, False)] for words in splited_setences]

# 行数：文章数、列数：単語個数に対応している
bag_of_words = np.zeros((len(splited_setences), n_word))
# それぞれの文で出現する単語に頻出回数ランキングの番号の列に1を代入
for i, numbers in enumerate(word_numbers):
    bag_of_words[i][numbers] = 1

#  行列化が完了したデータを保存
np.save(data_dir_path.joinpath('bag_of_words.npy'), bag_of_words)


#  アソシエーション分析のためのデータファイルを作成
output_file = [','.join(sentence) for sentence in splited_setences]
with open(data_dir_path.joinpath('data.basket'), 'w', encoding='utf-8') as file:
    file.writelines(output_file)

plt.show()