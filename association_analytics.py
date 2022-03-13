import shutil
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import Orange
from orangecontrib.associate.fpgrowth import *


data_dir_path = Path('.').joinpath('data')
result_dir_path = Path('.').joinpath('result')
image_dir_path = Path('.').joinpath('image')

if not result_dir_path.exists():
    result_dir_path.mkdir(parents=True)

if not image_dir_path.exists():
    image_dir_path.mkdir(parents=True)


##  共起ネットワークを表示する関数
def plot_network(data, edge_threshold=0., fig_size=(15, 15), file_name=None, dir_path=None):

    nodes = list(set(data['node1'].tolist() + data['node2'].tolist()))

    G = nx.DiGraph()
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
    pos = nx.spring_layout(G, k=0.5)  # k = node間反発係数

    pr = nx.pagerank(G)

    # nodeの大きさ
    nx.draw_networkx_nodes(G, pos, node_color=list(pr.values()),
                           cmap=plt.cm.Reds,
                           alpha=0.7,
                           node_size=[30000*v for v in pr.values()])

    # 日本語ラベル
    nx.draw_networkx_labels(G, pos, fontsize=14, font_family='IPAexGothic', font_weight="bold")

    # エッジの太さ調節
    edge_width = [d['weight']*2 for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="darkgrey", width=edge_width)

    plt.axis('off')

    if file_name is not None:
        if dir_path is None:
            dir_path = Path('.').joinpath('image')
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        plt.savefig(dir_path.joinpath(file_name), bbox_inches="tight")



def execute_association_analysis(file_name, lower_support, edge_threshold, data_dir_path, result_dir_path, fig_size=(12, 10)):

    #  単語の出現回数を計算
    data = Orange.data.Table(data_dir_path.joinpath(file_name).__str__())

    #  単語のインデックス化
    X, mapping = OneHot.encode(data)

    #  組み合わせの件数をカウントする
    #  第２引数のmin_supportで抽出するsupport(支持度）の最小値を指定できる
    #  組み合わせが3件以上のものが抽出される
    itemsets = dict(frequent_itemsets(X, lower_support))

    # アソシエーションルールの抽出
    rules = association_rules(itemsets, 0.3)

    # リフト値を含んだ結果を取得
    stats = rules_stats(rules, itemsets, len(X))

    def decode_onehot(d):
        items = OneHot.decode(d, data, mapping)
        return list(map(lambda v: v[1].name, items))

    # リフト値（7番目の要素）でソート
    result = []
    for s in sorted(stats, key=lambda x: x[6], reverse=True):
        lhs = decode_onehot(s[0])
        rhs = decode_onehot(s[1])

        support = s[2]
        confidence = s[3]
        lift = s[6]

        print("lhs = {}, rhs = {}, support = {}, confidence = {}, lift = {}".format(lhs, rhs, support, confidence, lift))
        result.append(['_'.join(lhs), '_'.join(rhs), support, confidence, lift])

    result = pd.DataFrame(result, columns=['lhs', 'rhs', 'support', 'confidence', 'lift'])
    result.to_csv(result_dir_path.joinpath(file_name.replace('splited_word.basket', 'association.csv')),
                  sep=',', index=False)

    result.rename(columns={'lhs':'node1', 'rhs':'node2', 'confidence':'value'}, inplace=True)
    plot_network(data=result,
                 edge_threshold=edge_threshold,
                 fig_size=fig_size,
                 file_name=file_name.replace('splited_word.basket', 'association_network.png'),
                 dir_path=image_dir_path)






if __name__ == '__main__':


    file_name = 'gakumonno_susume_splited_word.txt'
    basket_file_name = file_name.replace('.txt', '.basket')
    shutil.copy(data_dir_path.joinpath(file_name), data_dir_path.joinpath(basket_file_name))

    execute_association_analysis(file_name=basket_file_name,
                                 lower_support=6,
                                 edge_threshold=0.7,
                                 data_dir_path=data_dir_path,
                                 result_dir_path=result_dir_path,
                                 fig_size=(15, 15))

    plt.show()
