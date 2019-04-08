from pathlib import Path
import itertools
import collections
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from preprocessing import *
import seaborn as sns
pd.set_option('display.max_columns', 100)

data_dir_path = Path('.').joinpath('data')
result_dir_path = Path('.').joinpath('result')

if not result_dir_path.exists():
    result_dir_path.mkdir(parents=True)


##  共起ネットワークを表示する関数
def plot_network(data,edge_threshold=0., fig_size=(15, 15), file_name=None, dir_path=None):

    nodes = list(set(data['node1'] + data['node1']))

    G = nx.Graph()
    #  頂点の追加
    G.add_nodes_from(nodes)

    #  辺の追加
    #  edge_thresholdで枝の重みの下限を定めている
    for i in range(len(data)):
        row_data = data.iloc[i]
        if row_data['value'] > edge_threshold:
            G.add_edge(row_data['node1'], row_data['node2'], weight=row_data['value'])

    # 孤立したnodeを削除
    isolated = [n for n in G.nodes if len([i for i in nx.all_neighbors(G, n)]) == 0]
    for n in isolated:
        G.remove_node(n)

    plt.figure(figsize=fig_size)
    pos = nx.spring_layout(G, k=0.3)  # k = node間反発係数

    pr = nx.pagerank(G)

    # nodeの大きさ
    nx.draw_networkx_nodes(G, pos, node_color=list(pr.values()),
                           cmap=plt.cm.Reds,
                           alpha=0.7,
                           node_size=[60000*v for v in pr.values()])

    # 日本語ラベル
    nx.draw_networkx_labels(G, pos, fontsize=14, font_family='IPAexGothic', font_weight="bold")

    # エッジの太さ調節
    edge_width = [d["weight"] * 100 for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="darkgrey", width=edge_width)

    plt.axis('off')

    if file_name is not None:
        if dir_path is None:
            dir_path = Path('.').joinpath('image')
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        plt.savefig(dir_path.joinpath(file_name), bbox_inches="tight")


#  文章データから単語感の関連度を計算し、共起ネットワークを作る関数
def co_occurrence_network(sentences, n_word_lower=0, edge_threshold=0.0, fig_size=(),
                          word_jaccard_file_name=None, result_dir_path=None,
                          plot_file_name=None, plot_dir_path=None):

    sentence_combinations = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    sentence_combinations = [[tuple(sorted(words)) for words in sentence] for sentence in sentence_combinations]

    target_combinations = []
    for sentence in sentence_combinations:
        target_combinations.extend(sentence)

    ##------------------------------------  Jaccard係数を求める
    # Jaccard係数 = n(A ∩ B) / n(A ∪ B)

    #  直積の計算（同じ文内にある２つの単語の出現回数を計算）
    combi_count = count_word(target_combinations)

    word_associates = []
    for key, value in combi_count.items():
        word_associates.append([key[0], key[1], value])

    word_associates = pd.DataFrame(word_associates, columns=['word1', 'word2', 'intersection_count'])

    #  和集合の計算 n(A ∪ B) = n(A) + n(B) - n(A ∩ B) を利用
    #  それぞれの単語の出現回数を計算
    target_words = []
    for word in target_combinations:
        target_words.extend(word)

    word_and_count = count_word(target_words)
    word_and_count = [[key, value] for key, value in word_and_count.items()]
    word_and_count = pd.DataFrame(word_and_count, columns=['word', 'count'])

    #  単語の組合せの出現回数のデータにそれぞれの単語の出現回数を結合
    word_associates = pd.merge(word_associates, word_and_count, left_on='word1', right_on='word', how='left')
    word_associates.drop(columns=['word'], inplace=True)
    word_associates.rename(columns={'count': 'count1'}, inplace=True)
    word_associates = pd.merge(word_associates, word_and_count, left_on='word2', right_on='word', how='left')
    word_associates.drop(columns=['word'], inplace=True)
    word_associates.rename(columns={'count': 'count2'}, inplace=True)

    word_associates['union_count'] = word_associates['count1'] + word_associates['count2'] - word_associates['intersection_count']
    word_associates['jaccard_coefficient'] = word_associates['intersection_count'] / word_associates['union_count']
    word_associates.sort_values(by='jaccard_coefficient', ascending=False, inplace=True)

    if word_jaccard_file_name is not None:
        if result_dir_path is None:
            result_dir_path = Path('.').joinpath('result')
        if not result_dir_path.exists():
            result_dir_path.mkdir(parents=True)

        word_associates.to_csv(result_dir_path.joinpath(word_jaccard_file_name), encoding='utf-8', index=False)

    word_associates.query('count1 >= @n_word_lower & count2 >= @n_word_lower', inplace=True)
    word_associates.rename(columns={'word1':'node1', 'word2':'node2', 'jaccard_coefficient':'value'}, inplace=True)

    plot_network(data=word_associates, edge_threshold=edge_threshold, fig_size=fig_size, file_name=plot_file_name, dir_path=plot_dir_path)


## ------------------------ 共起ネットワーク構築関数を作るにあたっての実験
def experiment(base_file_name, sentences):

    #  単語の組合せを作っている
    sentence_combinations = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    sentence_combinations = [[tuple(sorted(words)) for words in sentence] for sentence in sentence_combinations]
    print('単語の組合せ')
    for combinations in sentence_combinations[:3]:
        print(combinations)
    print('')


    #  単語の組合せの1次元のリストに変形
    target_combinations = []
    for sentence in sentence_combinations:
        target_combinations.extend(sentence)

    ##------------------------------------  Jaccard係数を求める
    # Jaccard係数 = n(A ∩ B) / n(A ∪ B)

    #  直積の計算（同じ文内にある２つの単語の出現回数を計算）
    combi_count = collections.Counter(target_combinations)

    #  単語の組合せと出現回数のデータフレームを作る
    word_associates = []
    for key, value in combi_count.items():
        word_associates.append([key[0], key[1], value])

    word_associates = pd.DataFrame(word_associates, columns=['word1', 'word2', 'intersection_count'])

    #  和集合の計算 n(A ∪ B) = n(A) + n(B) - n(A ∩ B) を利用
    #  それぞれの単語の出現回数を計算
    target_words = []
    for word in target_combinations:
        target_words.extend(word)

    word_count = collections.Counter(target_words)
    word_count = [[key, value] for key, value in word_count.items()]
    word_count = pd.DataFrame(word_count, columns=['word', 'count'])

    #  単語の組合せの出現回数のデータにそれぞれの単語の出現回数を結合
    word_associates = pd.merge(word_associates, word_count, left_on='word1', right_on='word', how='left')
    word_associates.drop(columns=['word'], inplace=True)
    word_associates.rename(columns={'count': 'count1'}, inplace=True)
    word_associates = pd.merge(word_associates, word_count, left_on='word2', right_on='word', how='left')
    word_associates.drop(columns=['word'], inplace=True)
    word_associates.rename(columns={'count': 'count2'}, inplace=True)

    word_associates['union_count'] = word_associates['count1'] + word_associates['count2'] - word_associates['intersection_count']
    word_associates['jaccard_coefficient'] = word_associates['intersection_count'] / word_associates['union_count']

    print('Jaccard係数の算出')
    print(word_associates.head())

    word_associates.sort_values(by='jaccard_coefficient', ascending=False, inplace=True)

    jaccard_coefficients = word_associates['jaccard_coefficient']
    group_numbers = []
    for coefficient in jaccard_coefficients:
        if coefficient < 0.003:
            group_numbers.append(0)
        elif coefficient < 0.006:
            group_numbers.append(1)
        elif coefficient < 0.009:
            group_numbers.append(2)
        elif coefficient < 0.012:
            group_numbers.append(3)
        else:
            group_numbers.append(4)
    word_associates['group_number'] = group_numbers

    word_associates_group_sum = word_associates.groupby('group_number').count()
    word_associates_group_sum.reset_index(inplace=True)
    print(word_associates_group_sum.loc[:, ['group_number', 'word1']])
    print('')

    sns.pairplot(hue='group_number', data=word_associates.sample(800).loc[:, ['count1', 'count2', 'group_number']])
    plt.savefig(image_dir_path.joinpath(base_file_name+'_jaccard_group_plot.png'))

    group4 = word_associates.query('group_number == 4')
    group4.sort_values(by='jaccard_coefficient', ascending=False, inplace=True)
    print('Jaccard係数ランキングTOP１０')
    print(group4.head(10))



if __name__ == '__main__':

    base_file_name = 'gakumonno_susume'
    lines = read_data(base_file_name + '_cut_splited_word.txt', dir_path=data_dir_path)

    #  単語リストのテキストを単語に分割してリスト化し、見出しは省いている
    sentences = [line.replace('\n', '').split(',') for line in lines if not ('見出し' in line)]
    #  文内の単語が1語しかない場合は削除
    sentences = [sentence for sentence in sentences if len(sentence) > 1]

    #  データの調査のための関数
    experiment(base_file_name=base_file_name, sentences=sentences)


    #  共起ネットワーク構築の実行
    co_occurrence_network(sentences,
                          n_word_lower=150,
                          edge_threshold=0.01,
                          fig_size=(15, 13),
                          word_jaccard_file_name=base_file_name+'_word_combi_jaccard.csv',
                          result_dir_path=result_dir_path,
                          plot_file_name=base_file_name + '_co_occurrence_network.png',
                          plot_dir_path=image_dir_path)

    plt.show()



