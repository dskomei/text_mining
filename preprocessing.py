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



def read_data(file_name, dir_path=None):

    if dir_path is None:
        dir_path = Path('.')

    with open(dir_path.joinpath(file_name), 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.replace('\n', '') for line in lines]

    return lines


def write_data(data, file_name, dir_path=None):

    if dir_path is None:
        dir_path = Path('result')

    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    with open(dir_path.joinpath(file_name), 'w', encoding='utf-8') as file:
        file.writelines(data)


#  ストップワードリストを取得する関数
def get_stopword_lsit(file_name='stopword_list.txt', dir_path='.'):

    file_path = Path(dir_path).joinpath(file_name)
    if not file_path.exists():
        url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
        urllib.request.urlretrieve(url, file_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        stopword_list = [word.replace('\n', '') for word in file.readlines()]

    return stopword_list


def sentence_normalize(sentence):

    sentence = sentence.replace('\n', '')
    sentence = neologdn.normalize(sentence)
    sentence = unicodedata.normalize("NFKC", sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'[!-/:-@[-`{-~]', '', sentence)
    sentence = re.sub(r'\d+\.*\d*', '0', sentence)

    return sentence


def split_sentence(tagger, sentence, stopword_list=(), flag_unique=False):

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
    if flag_unique:
        words = list(set(words))

    return words


#  単語の出現回数を求める関数
def count_word(words):
    word_and_count = collections.Counter(words)
    return word_and_count


#  単語の出現回数データを単語と出現回数に分ける関数
def split_word_and_count(word_and_count, n_word=None):

    if n_word is None:
        n_word = len(word_and_count)

    #  単語の出現回数上位30までのグラフを作成
    word_count_most = word_and_count.most_common(n_word)
    words, values = [], []
    for word, value in word_count_most:
        words.append(word)
        values.append(value)

    return words, values


def get_splited_sentence(tagger, lines, stopword_list=(), sep='。', flag_unique=False):

    splited_setences = []
    for text in lines:
        texts = text.split(sep)
        for sentence in texts:
            word_lists = split_sentence(tagger=tagger, sentence=sentence, stopword_list=stopword_list, flag_unique=flag_unique)
            if len(word_lists) > 0:
                splited_setences.append(word_lists)

    return splited_setences


def plot_barh(xs, ys, figsize=(10, 6), save_file_name=None, save_dir_path=None):

    fig = plt.figure(figsize=figsize)
    plt.barh(xs, ys)
    fig.tight_layout()

    if save_file_name is not None:
        if save_dir_path is None:
            save_dir_path = Path('image')

            if not save_dir_path.exists():
                save_dir_path.mkdir(parents=True)

        plt.savefig(save_dir_path.joinpath(save_file_name), bbox_inches="tight")


#  Bag of Wordsを作る関数
def get_bag_of_words(splited_sentences, n_word=None, save_file_name=None, save_dir_path=None):

    splited_setences_1d = []
    for sentence in splited_sentences:
        splited_setences_1d.extend(sentence)

    word_and_count = count_word(words=splited_setences_1d)

    if n_word is None:
        n_word = len(word_and_count)


    #  出現回数が多い順に番号をふる
    word_dicts = dict([[key[0], i] for key, i in zip(word_and_count.most_common(n_word), range(n_word))])

    #  各文の単語を出現回数ランキング番号に変換する
    word_numbers = [[word_dicts[word] for word in words if word_dicts.get(word, False)] for words in splited_sentences]
    cut_splited_senteces = [[word for word in words if word_dicts.get(word, False)] for words in splited_sentences]
    cut_splited_senteces = [words for words in cut_splited_senteces if len(words) > 0]

    # 行数：文の数、列数：単語個数に対応している
    bag_of_words = np.zeros((len(splited_sentences), n_word))

    # それぞれの文で出現する単語に頻出回数ランキングの番号の列に1を代入
    for i, numbers in enumerate(word_numbers):
        for j in numbers:
            bag_of_words[i][j] = +1

    if save_file_name is not None:

        if save_dir_path is None:
            save_dir_path = Path('.').joinpath('data')
        if not save_dir_path.exists():
            save_dir_path.mkdir(parents=True)

        #  行列化が完了したデータを保存
        np.save(save_dir_path.joinpath(save_file_name), bag_of_words)

    return bag_of_words, cut_splited_senteces


if __name__ == '__main__':


    file_name = 'gakumonno_susume.txt'

    tagger = MeCab.Tagger("-Ochasen")
    tagger.parse('')

    lines = read_data(file_name=file_name, dir_path=data_dir_path)

    stopword_list = get_stopword_lsit()

    splited_sentences = get_splited_sentence(tagger=tagger, lines=lines, stopword_list=stopword_list, sep='。', flag_unique=True)

    splited_setences_1d = []
    for sentence in splited_sentences:
        splited_setences_1d.extend(sentence)

    word_and_count = count_word(words=splited_setences_1d)
    words, values = split_word_and_count(word_and_count=word_and_count, n_word=30)

    plot_barh(words, values, figsize=(10, 6), save_file_name='word_count_top30.png', save_dir_path=image_dir_path)

    n_word = len(word_and_count)
    words, values = split_word_and_count(word_and_count=word_and_count, n_word=n_word)
    print(word_and_count)
    print('文の数：{}, 単語数：{}'.format(len(splited_sentences), n_word))

    #  出現回数ごとの単語数を集計
    df = pd.DataFrame(word_and_count.most_common(n_word), columns=['word', 'count'])
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

    bag_of_words, cut_splited_sentences = get_bag_of_words(splited_sentences=splited_sentences,
                                                           n_word=n_word,
                                                           save_file_name='bag_of_words.npy',
                                                           save_dir_path=data_dir_path)

    #  出現回数による条件を満たした単語群の出力
    output_file = [','.join(sentence) + '\n' for sentence in cut_splited_sentences]
    write_data(data=output_file, file_name=file_name.replace('.txt', '_cut_splited_word.txt'), dir_path=data_dir_path)

    #  アソシエーション分析のためのデータファイルを作成
    output_file = [','.join(sentence)+'\n' for sentence in splited_sentences]
    write_data(data=output_file, file_name=file_name.replace('.txt', '_splited_word.txt'), dir_path=data_dir_path)

    plt.show()