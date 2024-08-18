import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FORMAT_WRITE = 'svg'
DIRS_WRITE = './figures/'
SAVE_FILE = False


def plot_potentials(t_eval,
                    potentials,
                    time_span=None,
                    title='',
                    save_file=SAVE_FILE):
    """複数の膜電位をプロットする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    potentials : np.ndarray or dict
        膜電位の値を保存した行列
        縦が時刻で横がユニット
    time_span : None or tuple of float
        キュー電流を流す時間の開始時刻と終了時刻
    title : str
        フィギュアのタイトル
    save_file : bool
        フィギュアを保存するかどうか
    """
    colors = ['tab:blue', 'tab:gray', 'tab:cyan', 'black']
    styles = ['solid', 'dashed', 'dotted', 'dashdot']

    if isinstance(potentials, dict):
        # 辞書型の場合はNumpy配列に変換
        potentials = [val for val in potentials.values()]
        potentials = np.array(potentials).T

    plt.figure(figsize=(12, 2))
    for idx in range(potentials.shape[1]):
        idx_plot = int(idx % len(colors))
        plt.plot(
            t_eval, potentials[:, idx],
            linewidth=1,
            linestyle=styles[idx_plot],
            color=colors[idx_plot],
        )
    if time_span is not None:
        for time_plot in time_span:
            plt.vlines(
                time_plot,
                np.min(potentials),
                np.max(potentials),
                colors='black',
                linestyles='dotted',
            )
    # plt.xlim(t_eval[0], t_eval[-1])
    plt.xlabel('時刻 [msec]')
    plt.ylabel('膜電位')
    plt.title(title)

    if save_file:
        plt.savefig(
            '{}{}.{}'.format(
                DIRS_WRITE,
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'),
                FORMAT_WRITE,
            ),
            format=FORMAT_WRITE,
        )
        plt.close()
    else:
        plt.show()


def plot_current_and_potential(t_eval,
                               current,
                               potential,
                               title='',
                               save_file=SAVE_FILE):
    """膜電位と電流を同時にプロットする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    current : np.ndarray or list
        電流の値を保存した一次元の配列
    potential : np.ndarray or list
        膜電位の値を保存した一次元の配列
    title : str
        フィギュアのタイトル
    save_file : bool
        フィギュアを保存するかどうか
    """
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(2, 1, 1)
    plt.plot(t_eval, current)
    plt.ylim(-5, 70)
    plt.ylabel('電流')

    plt.subplot(2, 1, 2)
    plt.plot(t_eval, potential)
    plt.ylim(-100, 70)
    plt.ylabel('膜電位')

    # plt.xlim(t_eval[0], t_eval[-1])
    plt.xlabel('時刻')
    fig.suptitle(title)
    fig.tight_layout()

    if save_file:
        plt.savefig(
            '{}{}.{}'.format(
                DIRS_WRITE,
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'),
                FORMAT_WRITE,
            ),
            format=FORMAT_WRITE,
        )
        plt.close()
    else:
        plt.show()


def plot_conductance(t_eval,
                     cond_diff,
                     conductances,
                     title='',
                     save_file=SAVE_FILE):
    """コンダクタンスをプロットする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    cond_diff : np.ndarray or list
        コンダクタンスの微分(f)の値を保存した一次元の配列
    conductance : np.ndarray or list
        コンダクタンス(g)の値を保存した一次元の配列
    title : str
        フィギュアのタイトル
    save_file : bool
        フィギュアを保存するかどうか
    """
    plt.figure(figsize=(12, 2))
    plt.plot(t_eval, cond_diff, label='dgdt (f)', color='black', linestyle='dashed')
    plt.plot(t_eval, conductances, label='コンダクタンス (g)', color='tab:blue', linestyle='solid')
    # plt.xlim(t_eval[0], t_eval[-1])
    plt.legend()
    plt.xlabel('時刻 [msec]')
    plt.ylabel('コンダクタンス')
    plt.title(title)

    if save_file:
        plt.savefig(
            '{}{}.{}'.format(
                DIRS_WRITE,
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'),
                FORMAT_WRITE,
            ),
            format=FORMAT_WRITE,
        )
        plt.close()
    else:
        plt.show()


def plot_raster(t_eval,
                spikes,
                time_span=None,
                title='',
                save_file=SAVE_FILE):
    """ラスタグラムをプロットする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    spikes : np.ndarray
        スパイク発火をゼロイチとして保存した行列
        縦が時刻で横がユニット数
    time_span : None or tuple of float
        キュー電流を流す時間の開始時刻と終了時刻
    title : str
        フィギュアのタイトル
    save_file : bool
        フィギュアを保存するかどうか
    """
    num_unit = spikes.shape[1]

    rslt = []
    for idx_unit in range(num_unit):
        rslt.extend([
            # 神経細胞の番号を1始まりに調整
            [t, idx_unit + 1] for t, spike in zip(t_eval, spikes[:, idx_unit])
            if spike == 1
        ])
    rslt = np.array(rslt).T

    plt.figure(figsize=(12, 4))
    if np.sum(rslt) != 0:
        plt.scatter(rslt[0], rslt[1], c='black', marker='s', s=0.6)
    else:
        # ひとつも発火がなかった場合の処理
        print('Raster plot is skipped because of no spikes.')
    plt.xlim(t_eval[0], t_eval[-1])
    plt.ylim(0, num_unit + 1)
    plt.xlabel('時刻 [msec]')
    plt.ylabel('ニューロンの番号')

    # 開始時刻に点線をつける
    if time_span is not None:
        for time_plot in time_span:
            plt.vlines(time_plot, -1, num_unit, colors='black', linestyles='dotted')
    plt.title(title)

    if save_file:
        plt.savefig(
            '{}{}.{}'.format(
                DIRS_WRITE,
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'),
                FORMAT_WRITE,
            ),
            format=FORMAT_WRITE,
        )
        plt.close()
    else:
        plt.show()



def highlight_connectivity(weight):
    """見やすくするために神経結合に強弱をつける
    """
    return 0.7 * weight

def calc_pos_x(degree):
    return np.cos(np.pi * ((- degree + 90) / 180))

def calc_pos_y(degree):
    return np.sin(np.pi * ((- degree + 90) / 180))


def make_jitter_x(df_node):
    split = np.where(df_node['exc'] == 1, df_node['idx'] % 4 + 1, 0)
    jitter = np.cos((- (split * 360 / 5) + 90) * (np.pi / 180))
    return jitter


def make_jitter_y(df_node):
    split = np.where(df_node['exc'] == 1, df_node['idx'] % 4 + 1, 0)
    jitter = np.sin((- (split * 360 / 5) + 90) * (np.pi / 180))
    return jitter


def format_architecture(architecture,
                        save_file=SAVE_FILE):
    """可視化用にネットワークアーキテクチャを辞書型からデータフレームに変換する

    Parameters
    ----------
    architecture : dict
        ワーキングメモリモデルのアーキテクチャ
        `generate_network_architecture()`の返り値

    Returns
    -------
    df_node : pd.DataFrame
        ネットワークの神経細胞に関する情報を保持
    df_edge : pd.DataFrame
        ネットワークの神経結合に関する情報を保持
    df_column : pd.DataFrame
        ネットワークのカラムに関する情報を保持
    save_file : bool
        フィギュアを保存するかどうか
    """
    scale = 0.08
    num_unit = architecture['num_unit']

    # 神経細胞
    df_node = pd.DataFrame({
        'idx': range(num_unit),
        'degree': architecture['positions'],
        'pos_x': calc_pos_x(np.array(architecture['positions'])),
        'pos_y': calc_pos_y(np.array(architecture['positions'])),
        'exc': architecture['excitation_binary'],
        'cue': architecture['positions_cue'],
    })
    # print(df_node.loc[:, ['pos_x']] + make_jitter_x(df_node=df_node) * scale)
    df_node['pos_x_jitter'] = df_node['pos_x'] + make_jitter_x(df_node=df_node) * scale
    df_node['pos_y_jitter'] = df_node['pos_y'] + make_jitter_y(df_node=df_node) * scale
    df_node['pos_x_jitter_large'] = df_node['pos_x'] + make_jitter_x(df_node=df_node) * scale * 1.62
    df_node['pos_y_jitter_large'] = df_node['pos_y'] + make_jitter_y(df_node=df_node) * scale * 1.62
    # df_node['pos_x_jitter_large'] = df_node['pos_x'] + make_jitter_x(df_node=df_node) * scale + 0.05
    # df_node['pos_y_jitter_large'] = df_node['pos_y'] + make_jitter_y(df_node=df_node) * scale

    # 神経結合
    weights = architecture['weights']
    list_weights = []
    for idx_row in range(num_unit):
        for idx_col in range(num_unit):
            if idx_col < idx_row:
                list_weights.append([
                    idx_row, idx_col, weights[idx_row, idx_col]
                ])
    df_edge = pd.DataFrame(
        list_weights,
        columns=['var_to', 'var_from', 'weight']
    )

    # カラム
    degrees = list(set(architecture['positions']))
    df_column = pd.DataFrame({
        'idx': range(len(degrees)),
        'degree': degrees,
        'pos_x': calc_pos_x(np.array(degrees)),
        'pos_y': calc_pos_y(np.array(degrees)),
    })

    return df_node, df_edge, df_column


def plot_network(architecture,
                 size_neuron=160,
                 size_column=3000,
                 display_circle=False,
                 display_weight=True,
                 display_index=False,
                 save_file=SAVE_FILE):
    """ネットワークアーキテクチャを可視化

    Parameters
    ----------
    architecture : dict
        ワーキングメモリモデルのアーキテクチャ
        `generate_network_architecture()`の返り値
    size_neuron : int
        表示される神経細胞（丸や三角）の大きさ
    size_column : int
        表示されるカラム（四角）の大きさ
    display_circle : bool
        円周をプロットするかどうか
    display_weight : bool
        重みづけ係数をプロットするかどうか
    display_index : bool
        神経細胞の番号をプロットするかどうか
    save_file : bool
        フィギュアを保存するかどうか
    """
    df_node, df_edge, df_column = format_architecture(architecture=architecture)
    num_unit = architecture['num_unit']

    plt.figure(figsize=(10, 10))

    # 円周のプロット
    if display_circle:
        degrees = np.linspace(0, 2 * np.pi, 10000)
        plt.plot(
            np.cos(degrees), np.sin(degrees),
            color='black',
            linestyle='dashed',
            linewidth=0.5,
            zorder=0,
        )

    # カラムのプロット
    plt.scatter(
        df_column.loc[:, ['pos_x']],
        df_column.loc[:, ['pos_y']],
        s=size_column,
        c='white',
        edgecolors='black',
        marker='s',
        linewidths=0.5,
    )

    # 神経結合
    if display_weight:
        for idx, weight in enumerate(df_edge['weight']):
            x_to = df_node['pos_x_jitter'][df_edge['var_to'][idx]]
            y_to = df_node['pos_y_jitter'][df_edge['var_to'][idx]]
            x_from = df_node['pos_x_jitter'][df_edge['var_from'][idx]]
            y_from = df_node['pos_y_jitter'][df_edge['var_from'][idx]]
            plt.plot(
                [x_to, x_from],
                [y_to, y_from],
                color='black',
                alpha=highlight_connectivity(weight),
                # linewidth=highlight_connectivity(weight),
                linewidth=0.2,
                zorder=1,
            )

    # 神経細胞のプロット
    # 興奮性細胞
    df_tmp = df_node.loc[df_node.loc[:, 'exc'] == 1, :]
    plt.scatter(
        df_tmp['pos_x_jitter'],
        df_tmp['pos_y_jitter'],
        s=size_neuron,
        c='white',
        edgecolors='blue',
        marker='^',
        linewidths=1.0,
        zorder=2,
    )

    # 抑制性細胞
    df_tmp = df_node.loc[df_node.loc[:, 'exc'] == 0, :]
    plt.scatter(
        df_tmp['pos_x_jitter'],
        df_tmp['pos_y_jitter'],
        s=size_neuron,
        c='white',
        edgecolors='blue',
        marker='o',
        linewidths=1.0,
        zorder=2,
    )

    # キュー電流
    df_tmp = df_node.loc[df_node.loc[:, 'cue'] == 1, :]
    plt.scatter(
        df_tmp['pos_x_jitter'],
        df_tmp['pos_y_jitter'],
        s=size_neuron,
        c='blue',
        edgecolors='blue',
        marker='^',
        linewidths=1.0,
        zorder=3,
    )

    # 神経細胞の番号
    if display_index:
        for idx in range(num_unit):
            plt.text(
                # 神経細胞の外側に表示する場合
                df_node['pos_x_jitter_large'][idx], df_node['pos_y_jitter_large'][idx],
                # 神経細胞の中心に表示する場合
                # df_node['pos_x_jitter'][idx], df_node['pos_y_jitter'][idx],
                '%d' % (int(df_node['idx'][idx] + 1)),
                size=7,
                color='black',
                horizontalalignment='center',
                verticalalignment='center',  # {'baseline', 'bottom', 'center', 'center_baseline', 'top'}
                zorder=4,
            )

    # カラムの位置（角度）
    for idx, degree in enumerate(df_column['degree']):
        plt.text(
            df_column['pos_x'][idx] + 0.09,
            df_column['pos_y'][idx] - 0.14,
            '%d' % (int(degree)),
            size=8,
            color='blue',
            horizontalalignment='center',
            verticalalignment='center',  # {'baseline', 'bottom', 'center', 'center_baseline', 'top'}
            zorder=4,
        )

    plt.axis("off")
    if save_file:
        plt.savefig(
            '{}{}.{}'.format(
                DIRS_WRITE,
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'),
                FORMAT_WRITE,
            ),
            format=FORMAT_WRITE,
        )
        plt.close()
    else:
        plt.show()


def plot_weight(architecture,
                save_file=SAVE_FILE):
    """重みづけ係数を可視化

    Parameters
    ----------
    architecture : dict
        ワーキングメモリモデルのアーキテクチャ
        `generate_network_architecture()`の返り値
    save_file : bool
        フィギュアを保存するかどうか
    """
    plt.figure(figsize=(7, 7))
    plt.pcolor(architecture['weights'], cmap=plt.cm.Blues)
    plt.xlabel("シナプス前細胞の番号")
    plt.ylabel("シナプス後細胞の番号")
    if save_file:
        plt.savefig(
            '{}{}.{}'.format(
                DIRS_WRITE,
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'),
                FORMAT_WRITE,
            ),
            format=FORMAT_WRITE,
        )
        plt.close()
    else:
        plt.show()
