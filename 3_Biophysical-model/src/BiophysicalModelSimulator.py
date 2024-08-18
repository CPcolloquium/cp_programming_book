import itertools

import matplotlib.pyplot as plt
import numpy as np  # 行列演算用ライブラリを読み込む
from scipy.integrate import solve_ivp

DELTA_T = 0.1  # ステップ幅

# 積分発火モデル
E_REST = -65.0  # 静止膜電位
V_ACT = 40.0  # 活動電位
V_RESET = -65.0  # リセット電位
V_INIT = -70.0  # 電位の初期値
V_THERESHOLD = -55.0  # 発火閾値

T_REF = 2.0  # 不応期 [ms]
T_DELAY = 2.0  # 発火の遅延 [ms]

# 時定数
TAU_1_AMPA, TAU_2_AMPA = 1.0, 5.0
TAU_1_NMDA, TAU_2_NMDA = 10.0, 100.0
TAU_1_GABA, TAU_2_GABA = 1.0, 5.0
TAU_CA = 5.0

# 静電容量（キャパシタンス）
C_LIF = 1.0
C_NMDA = 1.0 * 1000
C_EXC = 0.5 * 1000
C_INH = 0.2 * 1000

# ワーキングメモリモデル
E_LEAK = -70.0
E_AMPA = 0.0
E_NMDA = 0.0
E_GABA = -80.0

# ホジキン・ハックスリーモデル（関数内でも宣言）
E_REST_HH = -65.0

# 最大コンダクタンス値
G_MAX_REST = 1.0
G_MAX_LEAK = 25.0  # ワーキングメモリモデルで使用
G_MAX_LEAK_NMDA = 1.0  # NMDA受容体付き神経細胞で使用
G_MAX_LEAK_EXC = 25.0
G_MAX_LEAK_INH = 20.0
G_MAX_NA = 0.2
G_MAX_AMPA = 2.0
G_MAX_NMDA = 10.0
G_MAX_NMDA = 10.0 * 0.7
G_MAX_GABA = 10.0

# 積分発火モデルによる神経回路
G_MAX_LEAK_NETWORK = 1.0
E_LEAK_NETWORK = -65.0

# 注入電流（ワーキングメモリ）
CUE_SCALE = 7.0
CUE_SCALE_EXT = 0.5
CUE_SCALE_INH = 0.36

# 重みづけ係数（LIFベースのモデルを使用する場合）
WEIGHT_STRENGTH_E2E, WEIGHT_WIDTH_E2E = 0.8, 10
WEIGHT_STRENGTH_E2I, WEIGHT_WIDTH_E2I = 0.5, 50
WEIGHT_STRENGTH_I2E, WEIGHT_WIDTH_I2E = 0.5, 50
WEIGHT_STRENGTH_I2I, WEIGHT_WIDTH_I2I = 0.2, 40

# シードを固定し結果を再現可能に(実行順序にも依存するため要注意)
SEED = 42

# 評価時間の計算

def generate_t_eval(t_max, delta_t=DELTA_T):
    """評価時間を生成する
    Parameters
    ----------
    t_max : float
        最大時刻 (ミリ秒)
    delta_t : float
        ステップ幅 (ミリ秒)

    Returns
    -------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    """
    # 最大時刻から最大ステップ数を作成
    step_max = int(t_max / delta_t) + 1

    # 0からt_maxまでstep_max個に等分
    t_eval = np.linspace(0, t_max, step_max)
    return t_eval


# 微分方程式ソルバー

def solve_differential_equation(dydt, t, y, delta_t=1, method='euler', **kwargs):
    """微分方程式dydtを指定すると，次の時刻t+delta_tにおける状態yを返す

    Parameters
    ----------
    dydt : function
        引数yの微分を返す微分方程式
        dydt()は第一引数に時刻t, 第二引数に状態y, 可変長キーワード引数kwargsを持つ
    t : float
        時刻
    y : numpy.ndarray
        時刻tにおける状態
    delta_t : float
        ステップ幅 (ミリ秒)
    method : str
        'euler'の場合オイラー法，'rk4'の場合ルンゲ・クッタ法を使用
    kwargs : dict
        可変長キーワード引数。この値は，微分方程式dydtにそのまま渡される
        そのため，時刻や状態以外の変数をdydtに与えたい際に利用する

    Returns
    -------
    y_next : numpy.ndarray
        時刻t + delta_tにおける状態yの値
    """
    if method == 'euler':
        y_next = solve_differential_equation_euler(
            dydt=dydt,
            t=t,
            y=y,
            delta_t=delta_t,
            **kwargs
        )
    elif method == 'rk4':
        y_next = solve_differential_equation_rk4(
            dydt=dydt,
            t=t,
            y=y,
            delta_t=delta_t,
            **kwargs
        )
    else:
        raise NotImplementedError()
    return y_next

def solve_differential_equation_euler(dydt, t, y, delta_t=1, **kwargs):
    """オイラー法を用いて次の時刻t+delta_tにおける状態yを返す
    """
    y_next = delta_t * dydt(t, y, **kwargs) + y
    return y_next

def solve_differential_equation_rk4(dydt, t, y, delta_t=1, **kwargs):
    """ルンゲ・クッタ法を用いて次の時刻t+delta_tにおける状態yを返す
    """
    h = delta_t
    k_1 = h * dydt(t, y, **kwargs)
    k_2 = h * dydt(t + 0.5 * h, y + 0.5 * k_1, **kwargs)
    k_3 = h * dydt(t + 0.5 * h, y + 0.5 * k_2, **kwargs)
    k_4 = h * dydt(t + h, y + k_3, **kwargs)
    y_next = y + (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    return y_next


# 積分発火モデル

def calc_lif(t, potential, current_ext, last_spike):
    """1時刻における積分発火モデルの計算

    オイラー法を用いて1時刻における積分発火モデルにおける膜電位の計算を行う

    Parameters
    ----------
    t : float
        時刻 (ms)
    potential : float
        膜電位
    current_ext :
        注入電流
    last_spike :
        直近の発火時刻

    Returns
    -------
    potential_next : float
        次の時刻における膜電位
    """
    if last_spike < t and t <= last_spike + T_REF:
        # 不応期の間は膜電位をリセット電位に固定
        # 式(3-1)の第三式に対応
        potential_next = V_RESET
    elif potential >= V_THERESHOLD:
        # 不応期ではなく，かつ発火閾値に達した場合は
        # 活動電位に固定。式(3-1)の第二式に対応
        potential_next = V_ACT
    else:
        # 不応期ではなく，かつ発火閾値に達しない場合は
        # 微分方程式を用いた更新
        # 式(3-1)の第一式や式(3-2)に対応
        potential_delta = DELTA_T * (1.0 / C_LIF) * (
            - G_MAX_REST * (potential - E_REST) + current_ext)
        potential_next = potential + potential_delta
    return potential_next


def simulate_lif(t_eval,
                 current_max=20.0):
    """外部電流に対するLIFモデルの振る舞いをシミュレーション

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    current_max : float
        注入電流の最大値

    Returns
    -------
    potential : np.ndarray
        膜電位の時系列全体
    """
    # [B] 初期値の設定
    potential = V_INIT
    current = 0
    spike = 0
    last_spike = -100

    # [C] 結果保存用変数の準備
    results = {
        'potential': [],
        'current': [],
        'spike': [],
    }

    for t in t_eval:
        # [D] 計算結果を保存
        results['potential'].append(potential)
        results['current'].append(current)
        results['spike'].append(spike)

        # [E] 各時刻における計算
        # [E-a] 各時刻における注入電流の作成
        if t < 30.0:
            current = 0
        elif t < 60.0:
            # 30から60ミリ秒にかけて徐々に電流は増加
            current = (current_max / 30.0) * (t - 30.0)
        elif t < 90.0:
            current = current_max
        else:
            current = 0

        # [E-b] 各時刻における膜電位の更新
        potential = calc_lif(
            t=t,
            potential=potential,
            current_ext=current,
            last_spike=last_spike,
        )

        # [E-c] スパイクの判定
        spike = 1 if potential >= V_ACT else 0

        # [E-d] 直近の発火時刻の更新
        last_spike = t if spike == 1 else last_spike

    return results


# 積分発火モデルを用いた回路設計


def differentiate_network(t, y, **kwargs):
    """3つの積分発火モデルにおける微分を返す

    Parameters
    ----------
    t : float
        時刻
    y : np.ndarray
        3つの膜電位
    kwargs : dict
        可変長キーワード引数

    Returns
    -------
    dydt : np.ndarray
        3つの膜電位に関する微分値
    """
    weights = kwargs['weights']
    spikes = kwargs['spikes']
    cue_scale = kwargs['cue_scale']

    # [E-a-i] 注入電流
    # 今回は簡便に微分方程式内部でノイズを発生
    currents_cue = cue_scale * np.random.rand(*y.shape)

    # [E-a-ii] シナプスからの流入電流
    currents_syn = np.matmul(weights, spikes)

    # [E-a-iii] 神経細胞全体の流入電流
    currents_ext = currents_cue + currents_syn

    # [E-a-iv] 微分方程式の計算
    dydt = - G_MAX_LEAK_NETWORK * (y - E_LEAK_NETWORK) + currents_ext
    return dydt


def simulate_network(t_eval,
                     weight):
    """3つの積分発火モデルをつなげたネットワークをシミュレーションする

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    weight : float
        神経細胞間の結合強度。今回のモデルでは，すべての結合は一定

    Returns
    -------
    potential : np.ndarray
        膜電位 () をシーケンス

    Notes
    -----
    オイラー法を前提に実装
    """
    # [A] シミュレーションの設定
    np.random.seed(SEED)  # シードを固定し実行ごとの乱数を統一する

    # 注入電流の大きさ
    cue_scale = 20
    # 神経細胞数は3で固定
    num_unit = 3
    # 重みづけ係数は，値がweightのnum_unit x num_unitの行列とする
    weights = weight * np.ones((num_unit, num_unit))

    # [B] 初期値の設定
    potentials = V_INIT * np.ones((num_unit, 1))
    spikes = np.zeros((num_unit, 1))
    last_spikes = -100 * np.ones((num_unit, 1))

    # [C] 結果保存用変数の準備
    results = {
        'potentials': [],
        'spikes': [],
    }

    for t in t_eval:
        # [D] 計算結果を保存
        results['potentials'].append(potentials)
        results['spikes'].append(spikes)

        # [E] 各時刻における計算
        # [E-a] 微分方程式による更新
        potentials = solve_differential_equation(
            dydt=differentiate_network,
            t=t,
            y=potentials,
            delta_t=DELTA_T,
            method='euler',
            spikes=spikes,
            weights=weights,
            cue_scale=cue_scale,
        )

        # [E-b] 積分発火モデルによる更新
        refractory = (last_spikes < t) & (t <= last_spikes + T_REF)
        active = (potentials >= V_THERESHOLD) & (~ refractory)
        potentials[active] = V_ACT
        potentials[refractory] = V_RESET

        # [E-c] スパイクの判定
        spikes = np.where(potentials == V_ACT, 1, 0)

        # [E-d] 直近の発火時刻の更新
        last_spikes = np.where(
            spikes == 1, t, last_spikes
        )

    # [F] 返り値の準備
    for key, value in results.items():
        # 結果の値をNumpy型に整形
        results[key] = np.hstack(value).T

    return results


# コンダクタンスのモデル
def calc_synaptic_effect(last_spikes,
                         t,
                         weights,
                         g_max):
    """シナプスの影響度を演算

    式(3-3)に相当

    Parameters
    ----------
    last_spikes : np.ndarray
        直近に発火をした時刻
        次元は，シナプス前細胞の数 x 1
    t : float
        時刻
    weights : np.ndarray
        重みづけ係数
        次元は，シナプス後細胞の数 x シナプス前細胞の数
    g_max : float
        最大コンダクタンス値

    Returns
    -------
    e : np.ndarray
        シナプスの影響度
        次元は，シナプス後細胞の数 x シナプス前細胞の数

    Notes
    -----
    行列の計算に注意！
    """
    # 直近の時刻から，シナプス前細胞におけるスパイクを判定
    # 配列の形状: num_unit_pre x 1
    spikes_pre = np.where(
        (t - 2 * DELTA_T < last_spikes + T_DELAY) & (last_spikes + T_DELAY <= t - DELTA_T),
        1.0, 0.0
    )

    # スパイク発火の行列を整形
    # 配列の形状: num_unit_post x num_unit_pre
    spikes_pre = np.tile(spikes_pre.T, reps=(weights.shape[0], 1))

    # 配列の形状: num_unit_post x num_unit_pre
    e = g_max * weights * (spikes_pre / DELTA_T)
    return e


def calc_dfdt(f, g, e, tau_1, tau_2):
    """コンダクタンスの二階微分を計算

    式(3-4)に相当

    Parameters
    ----------
    f : np.ndarray
        コンダクタンスの微分
        次元は，シナプス後細胞の数 x シナプス前細胞の数
    g : np.ndarray
        コンダクタンス
        次元は，シナプス後細胞の数 x シナプス前細胞の数
    e : np.ndarray
        前シナプスからの影響度
        次元は，シナプス後細胞の数 x シナプス前細胞の数

    Returns
    -------
    dfdt : np.ndarray
        コンダクタンスの二階微分
        次元は，シナプス後細胞の数 x シナプス前細胞の数
    """
    tau_sumprod = (1 / tau_1) + (1 / tau_2)
    dfdt = - tau_sumprod * f \
        - (1 / (tau_1 * tau_2)) * g \
        + tau_sumprod * e
    return dfdt


def differentiate_channel(t, y, **kwargs):
    """コンダクタンスの微分と二階微分を返す

    Parameters
    ----------
    t : float
        時刻
    y : np.ndarray
        コンダクタンスgとコンダクタンスの微分fを合体した変数
    kwargs : dict
        可変長キーワード引数

    Returns
    -------
    dydt : np.ndarray
        コンダクタンスの微分dgdtと二階微分dfdtを合体した変数
    """
    # 変数の取得
    last_spikes = kwargs['last_spikes']
    weights = kwargs['weights']
    tau_1 = kwargs['tau_1']
    tau_2 = kwargs['tau_2']

    # 状態yをコンダクタンスの微分fとコンダクタンスgに分解
    f_receptor, g_receptor = np.split(y, [1], axis=1)

    # [E-a-i] シナプスの影響度eの計算
    # 式(3-3)に相当
    e_receptor = calc_synaptic_effect(
        last_spikes=last_spikes,
        t=t,
        weights=weights,
        g_max=1.0,
    )

    # [E-a-ii] コンダクタンスの微分fの計算（微分）
    # 式(3-4)に相当
    dfdt = calc_dfdt(
        f=f_receptor,
        g=g_receptor,
        e=e_receptor,
        tau_1=tau_1,
        tau_2=tau_2,
    )

    # [E-a-iii] コンダクタンスgの計算（微分）
    # 式(3-5)に相当
    dgdt = f_receptor

    # 微分の値を一つの変数にまとめる
    dydt = np.hstack([dfdt, dgdt])
    return dydt


def simulate_channel(t_eval, tau_1, tau_2):
    """コンダクタンスの挙動をシミュレーションする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    tau_1 : float
        コンダクタンスモデルに渡される時定数tau_1
    tau_2 : float
        コンダクタンスモデルに渡される時定数tau_2

    Returns
    -------
    results : dict
        コンダクタンスやその微分の変化をリストとして値に保存した辞書
    """
    # [A] シミュレーションの設定
    # 重みづけ係数は，値が1の1 x 1の行列とする.
    # 神経細胞数は1
    weights = np.ones((1, 1))

    # [B] 初期値の設定
    last_spikes = -100 * np.ones((1, 1))
    f_receptor = np.zeros((1, 1))
    g_receptor = np.zeros((1, 1))

    # 変数を一つにまとめる
    y = np.hstack([f_receptor, g_receptor])

    # [C] 結果保存用変数の準備
    results = {
        'f_receptor': [],
        'g_receptor': [],
    }

    for t in t_eval:
        # [D] 計算結果を保存
        results['f_receptor'].append(f_receptor)
        results['g_receptor'].append(g_receptor)

        # [E] 各時刻における計算
        # [E-a] 微分方程式による更新
        y = solve_differential_equation(
            dydt=differentiate_channel,
            t=t,
            y=y,
            delta_t=DELTA_T,
            method='rk4',
            last_spikes=last_spikes,
            weights=weights,
            tau_1=tau_1,
            tau_2=tau_2,
        )
        f_receptor, g_receptor = np.split(y, [1], axis=1)

        # [E-b] スパイクの判定
        # 10-20msecは常時発火と仮定
        spikes = np.where(
            10 < t and t < 20 , 1, 0
        ).reshape(1, 1)

        # [E-c] 直近の発火時刻の更新
        last_spikes = np.where(
            spikes == 1, t, last_spikes
        )

    # [F] 返り値の準備
    for key, value in results.items():
        # 結果の値をNumpy型に整形
        results[key] = np.hstack(value).T
    return results


# NMDA受容体付き神経細胞
def calc_block_mg(potential):
    """NMDA受容体のマグネシウムブロックの値を計算
    """
    # 式(3-9)に相当
    return 1 / (1 + 0.5 * np.exp(- 0.062 * potential))


def differentiate_nmda_unit(t, y, **kwargs):
    """コンダクタンスの微分と二階微分を返す

    Parameters
    ----------
    t : float
        時刻
    y : np.ndarray
        コンダクタンスとコンダクタンスの微分を合体した変数
    kwargs : dict
        可変長キーワード引数

    Returns
    -------
    dydt : np.ndarray
        コンダクタンスの微分と二階微分を合体した変数
    """
    # 変数の取得
    last_spikes = kwargs['last_spikes']
    weights = kwargs['weights']
    tau_1 = kwargs['tau_1']
    tau_2 = kwargs['tau_2']

    # 状態yをコンダクタンスの微分f，コンダクタンスg，膜電位vに分解
    f_nmda, g_nmda, potentials = np.split(y, [1, 2], axis=1)

    # [E-a-i] シナプスの影響度eの計算
    # 式(3-3)に相当
    e_nmda = calc_synaptic_effect(
        last_spikes=last_spikes,
        t=t,
        weights=weights,
        g_max=G_MAX_NMDA,
    )

    # [E-a-ii] コンダクタンスの微分fの計算（微分）
    # 式(3-4)に相当
    dfdt = calc_dfdt(
        f=f_nmda,
        g=g_nmda,
        e=e_nmda,
        tau_1=tau_1,
        tau_2=tau_2,
    )

    # [E-a-iii] コンダクタンスgの計算（微分）
    # 式(3-5)に相当
    dgdt = f_nmda

    # [E-a-iv] 電流Iの計算
    # 式(3-9)に相当
    block_mg = calc_block_mg(potentials)
    # 式(3-8)に相当
    currents_nmda = - block_mg * g_nmda * (potentials - E_NMDA)
    # 式(3-7)に相当
    currents_leak = - G_MAX_LEAK_NMDA * (potentials - E_LEAK)
    # 100から150ミリ秒に注入電流を与える
    currents_cue = 1000.0 if t >= 100.0 and t < 150.0 else 0.0
    # 式(3-6)に相当
    currents = currents_nmda + currents_leak + currents_cue

    # [E-a-v] 膜電位vの計算（微分）
    # 式(3-10)に相当
    dvdt = currents / C_NMDA

    # 微分の値を一つの変数にまとめる
    dydt = np.hstack([dfdt, dgdt, dvdt])
    return dydt


def simulate_nmda_unit(t_eval,
                       weight=4.0,
                       tau_1=10.0,
                       tau_2=100.0):
    """コンダクタンスの挙動をシミュレーションする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    weight : float
        重みづけ係数
    tau_1 : float
        コンダクタンスモデルに渡される時定数tau_1
    tau_2 : float
        コンダクタンスモデルに渡される時定数tau_2

    Returns
    -------
    results : dict
        コンダクタンスやその微分の変化をリストとして値に保存した辞書
    """
    # [A] シミュレーションの設定
    # 重みづけ係数は，値が1の1 x 1の行列とする. 神経細胞数は1
    weights = weight * np.ones((1, 1))

    # [B] 初期値の設定
    last_spikes = -100 * np.ones((1, 1))
    f_nmda = np.zeros((1, 1))
    g_nmda = np.zeros((1, 1))
    potentials = V_INIT * np.ones((1, 1))

    # [C] 結果保存用変数の準備
    results = {
        'f_nmda': [],
        'g_nmda': [],
        'potentials': [],
    }

    for t in t_eval:
        # [D] 計算結果を保存
        results['f_nmda'].append(f_nmda)
        results['g_nmda'].append(g_nmda)
        results['potentials'].append(potentials)

        # [E] 各時刻における計算
        # 変数を一つにまとめる
        y = np.hstack([f_nmda, g_nmda, potentials])

        # [E-a] 微分方程式による更新
        y = solve_differential_equation(
            dydt=differentiate_nmda_unit,
            t=t,
            y=y,
            delta_t=DELTA_T,
            method='rk4',
            last_spikes=last_spikes,
            weights=weights,
            tau_1=tau_1,
            tau_2=tau_2,
        )
        f_nmda, g_nmda, potentials = np.split(y, [1, 2], axis=1)

        # [E-b] 積分発火モデルによる更新
        refractory = (last_spikes < t) & (t <= last_spikes + T_REF)
        active = (potentials >= V_THERESHOLD) & (~ refractory)
        potentials[active] = V_ACT
        potentials[refractory] = V_RESET

        # [E-c] スパイクの判定
        spikes = np.where(potentials == V_ACT, 1, 0)

        # [E-d] 直近の発火時刻の更新
        last_spikes = np.where(spikes == 1, t, last_spikes)

    # [F] 返り値の準備
    for key, value in results.items():
        # 結果の値をNumpy型に整形
        results[key] = np.hstack(value).T

    return results


# ワーキングメモリモデル
def calc_exp(dist, strength, width):
    """距離から重みづけ係数を計算
    Parameters
    ----------
    dist
        細胞間の距離
    strength
        結合強度の分布の最大値を決める
    width
        結合強度の分布の横幅を決める

    Returns
    -------
        重みづけ係数
    """
    # 式(3-11)に相当
    w = strength * np.exp(- (dist**2) / (2 * width**2))
    return w


def calc_weights(strength, width, degree_pre, degree_post):
    """シナプス前・後細胞の位置から距離を計算し重みづけ係数を計算
    """
    # 配列の形状: num_unit_post x num_unit_pre
    degree_pre_mat = np.vstack([degree_pre for theta in degree_post])

    # 配列の形状: num_unit_post x num_unit_pre
    degree_post_mat = np.vstack([degree_post for theta in degree_pre]).T

    # シナプス前細胞と後細胞の距離を計算
    # 配列の形状: num_unit_post x num_unit_pre
    dist = np.minimum(
        (degree_pre_mat - degree_post_mat) % 360,
        (degree_post_mat - degree_pre_mat) % 360,
    )
    weights = calc_exp(
        dist=dist,
        strength=strength,
        width=width,
    )
    return weights


def init_weights_circle(set_exc,
                        set_inh,
                        positions,
                        ):
    """円形に配置されたニューロン間の重みを作成
    """
    num_unit = len(positions)
    num_unit_exc = len(set_exc)
    num_unit_inh = int(num_unit * 0.2)  # 抑制性神経細胞の総数
    num_column = num_unit_inh  # カラムの総数

    strength_exc2exc, width_exc2exc = WEIGHT_STRENGTH_E2E, WEIGHT_WIDTH_E2E
    strength_exc2inh, width_exc2inh = WEIGHT_STRENGTH_E2I, WEIGHT_WIDTH_E2I
    strength_inh2exc, width_inh2exc = WEIGHT_STRENGTH_I2E, WEIGHT_WIDTH_I2E
    strength_inh2inh, width_inh2inh = WEIGHT_STRENGTH_I2I, WEIGHT_WIDTH_I2I

    # weights = np.empty(shape=(num_unit, num_unit))
    weights = 100 * np.ones(shape=(num_unit, num_unit))
    # weights_from_exc = np.empty(shape=(num_unit, num_unit_exc))
    # weights_from_inh = np.empty(shape=(num_unit, num_unit_inh))

    # 興奮性細胞から興奮性細胞の結合
    weights[np.ix_(set_exc, set_exc)] = calc_weights(
        strength=strength_exc2exc,
        width=width_exc2exc,
        degree_pre=np.array(positions)[set_exc],
        degree_post=np.array(positions)[set_exc],
    )
    # weights_from_exc[set_exc, :] = calc_weights(
    #     strength=strength_exc2exc,
    #     width=width_exc2exc,
    #     degree_pre=np.array(positions)[set_exc],
    #     degree_post=np.array(positions)[set_exc],
    # )

    # 興奮性細胞から抑制性細胞の結合
    weights[np.ix_(set_inh, set_exc)] = calc_weights(
        strength=strength_exc2inh,
        width=width_exc2inh,
        degree_pre=np.array(positions)[set_exc],
        degree_post=np.array(positions)[set_inh],
    )
    # weights_from_exc[set_inh, :] = calc_weights(
    #     strength=strength_exc2inh,
    #     width=width_exc2inh,
    #     degree_pre=np.array(positions)[set_exc],
    #     degree_post=np.array(positions)[set_inh],
    # )

    # 抑制性細胞から興奮性細胞の結合
    weights[np.ix_(set_exc, set_inh)] = calc_weights(
        strength=strength_inh2exc,
        width=width_inh2exc,
        degree_pre=np.array(positions)[set_inh],
        degree_post=np.array(positions)[set_exc],
    )
    # weights_from_inh[set_exc, :] = calc_weights(
    #     strength=strength_inh2exc,
    #     width=width_inh2exc,
    #     degree_pre=np.array(positions)[set_inh],
    #     degree_post=np.array(positions)[set_exc],
    # )

    # 抑制性細胞から抑制性細胞の結合
    weights[np.ix_(set_inh, set_inh)] = calc_weights(
        strength=strength_inh2inh,
        width=width_inh2inh,
        degree_pre=np.array(positions)[set_inh],
        degree_post=np.array(positions)[set_inh],
    )
    # weights_from_inh[set_inh, :] = calc_weights(
    #     strength=strength_inh2inh,
    #     width=width_inh2inh,
    #     degree_pre=np.array(positions)[set_inh],
    #     degree_post=np.array(positions)[set_inh],
    # )
    return weights


def generate_network_architecture(num_unit,
                                  ion=False):
    # 神経細胞の個数の設定
    num_unit_exc = int(num_unit * 0.8)  # 興奮性神経細胞の総数
    num_unit_inh = int(num_unit * 0.2)  # 抑制性神経細胞の総数
    num_column = num_unit_inh  # カラムの総数

    # np.split()で使用するインデックスのリストを用意
    split_list = [
        # AMPA受容体のコンダクタンス
        num_unit_exc,
        # AMPA受容体のコンダクタンスの微分
        2 * num_unit_exc,
        # NMDA受容体のコンダクタンス
        3 * num_unit_exc,
        # NMDA受容体のコンダクタンスの微分
        4 * num_unit_exc,
        # GABA受容体のコンダクタンス
        4 * num_unit_exc + num_unit_inh,
        # GABA受容体のコンダクタンスの微分
        4 * num_unit_exc + 2 * num_unit_inh,
    ]
    if ion:
        # カルシウム濃度の数を追加する
        split_list = split_list + [
            # 膜電位
            4 * num_unit_exc + 2 * num_unit_inh + 1,
        ]

    # 興奮性・抑制性神経細胞の設定
    set_exc = [  # 興奮性を示す添字集合
        idx for idx in range(num_unit) if idx < num_unit_exc
    ]
    set_inh = [  # 抑制性を示す添字集合
        idx for idx in range(num_unit) if idx >= num_unit_exc
    ]
    excitation_binary = [  # 興奮性の場合1を持つリストも用意
        1 if idx in set_exc else 0 for idx in range(num_unit)
    ]

    # 神経細胞に対する位置（角度）の割り当て
    positions_exc = [  # 興奮性細胞に位置を割り当て
        np.floor(idx / 4) * np.floor(360 / num_column) \
        for idx in range(num_unit_exc)
    ]
    positions_inh = [  # 抑制性細胞に位置を割り当てる
        idx * np.floor(360 / num_column) \
        for idx in range(num_unit_inh)
    ]
    # 興奮性細胞と抑制性細胞の位置を結合
    positions = positions_exc + positions_inh


    # キュー電流を与える神経細胞の割り当て
    # 角度が180かつ興奮性の細胞にのみキュー電流を与える
    positions_cue = np.where(
        (np.array(positions) == 180) \
            & (np.array(excitation_binary) == 1), 1, 0)

    # 重み
    weights = init_weights_circle(
        set_exc=set_exc,
        set_inh=set_inh,
        positions=positions,
    )

    architecture = {
        'num_unit': num_unit,
        'num_unit_exc': num_unit_exc,
        'num_unit_inh': num_unit_inh,
        'split_list': split_list,
        'set_exc': set_exc,
        'set_inh': set_inh,
        'excitation_binary': excitation_binary,
        'positions': positions,
        'positions_cue': positions_cue,
        'weights': weights,
        # 'weights_from_exc': weights_from_exc,
        # 'weights_from_inh': weights_from_inh,
    }

    return architecture


def test_weight():
    # 重みづけ係数の算出
    weights_from_exc, weights_from_inh = bms.init_weights_circle(
        set_exc=set_exc,
        set_inh=set_inh,
        positions=positions,
    )

    # 結果を確認すると意図通りのサイズになっていることが分かる
    print(
        'Shape of weights_from_exc:', weights_from_exc.shape,
        'Shape of weights_from_inh:', weights_from_inh.shape,
    )

    # 念の為値も確認してみる
    print('weights_from_exc:', weights_from_exc[:10, :10])


def differentiate_working_memory(t, y, **kwargs):
    """ワーキングメモリモデルの微分方程式

    コンダクタンスの微分p, コンダクタンスg, 膜電位vについて微分する
    状態yは，上記の変数を保存

    Parameters
    ----------
    t: float
        時刻
    y : np.ndarray
        NMDA/AMPA/GABA受容体のコンダクタンスの微分p, コンダクタンスg, 膜電位v
    kwargs : dict
        ノイズ電流currents_noiseと直近のスパイク時刻last_spikesを含むこと

    Returns
    -------
    dydt : np.ndarray
        コンダクタンスの微分p, コンダクタンスg, 膜電位vに関する微分
    """
    # 引数の取得
    currents_noise = kwargs['currents_noise']
    last_spikes = kwargs['last_spikes']
    dysfuncs_ampa = kwargs['dysfuncs_ampa']
    dysfuncs_nmda = kwargs['dysfuncs_nmda']
    dysfuncs_gaba = kwargs['dysfuncs_gaba']

    architecture = kwargs['architecture']
    set_exc = architecture['set_exc']
    set_inh = architecture['set_inh']

    (f_ampa, g_ampa,
     f_nmda, g_nmda,
     f_gaba, g_gaba,
     potentials, ) = np.split(
        y, architecture['split_list'], axis=1)

    # [E-a-i] シナプスの影響度eの計算
    # 式(3-3)に相当
    e_ampa = calc_synaptic_effect(
        last_spikes=last_spikes[set_exc, :],
        t=t,
        weights=architecture['weights'][:, set_exc],
        g_max=G_MAX_AMPA * dysfuncs_ampa,
    )
    e_nmda = calc_synaptic_effect(
        last_spikes=last_spikes[set_exc, :],
        t=t,
        weights=architecture['weights'][:, set_exc],
        g_max=G_MAX_NMDA * dysfuncs_nmda,
    )
    e_gaba = calc_synaptic_effect(
        last_spikes=last_spikes[set_inh, :],
        t=t,
        weights=architecture['weights'][:, set_inh],
        g_max=G_MAX_GABA * dysfuncs_gaba,
    )

    # [E-a-ii] コンダクタンスの微分fの計算（微分）
    # 式(3-4)に相当
    dfdt_ampa = calc_dfdt(
        f=f_ampa,
        g=g_ampa,
        e=e_ampa,
        tau_1=TAU_1_AMPA,
        tau_2=TAU_2_AMPA,
    )
    dfdt_nmda = calc_dfdt(
        f=f_nmda,
        g=g_nmda,
        e=e_nmda,
        tau_1=TAU_1_NMDA,
        tau_2=TAU_2_NMDA,
    )
    dfdt_gaba = calc_dfdt(
        f=f_gaba,
        g=g_gaba,
        e=e_gaba,
        tau_1=TAU_1_GABA,
        tau_2=TAU_2_GABA,
    )

    # [E-a-iii] コンダクタンスgの計算（微分）
    # 式(3-5)に相当
    dgdt_ampa = f_ampa
    dgdt_nmda = f_nmda
    dgdt_gaba = f_gaba

    # [E-a-iv] 電流の計算
    # 式(3-14)に相当
    currents_ampa = - np.sum(
        g_ampa * np.tile(
            potentials - E_AMPA, len(set_exc)),
        axis=1,
        keepdims=True,
    )
    block_mg = calc_block_mg(potentials)
    currents_nmda = - np.sum(
        g_nmda * np.tile(
            block_mg * (potentials - E_NMDA), len(set_exc)),
        axis=1,
        keepdims=True,
    )
    currents_gaba = - np.sum(
        g_gaba * np.tile(
            potentials - E_GABA, len(set_inh)),
        axis=1,
        keepdims=True,
    )
    currents_leak = np.empty_like(currents_nmda)
    currents_leak[set_exc, :] = \
        - G_MAX_LEAK_EXC * (potentials[set_exc, :] - E_LEAK)
    currents_leak[set_inh, :] = \
        - G_MAX_LEAK_INH * (potentials[set_inh, :] - E_LEAK)

    # キュー電流の計算。100-200m秒で電流を流す
    currents_cue = \
        (C_EXC if 100.0 < t and t <= 200.0 else 0.0) \
        * architecture['positions_cue'].reshape(-1, 1)

    # それぞれの電流をまとめる
    # 式(3-12)に相当
    currents = \
        currents_ampa + currents_nmda + currents_gaba \
        + currents_leak + currents_cue + currents_noise

    # [E-a-v] 膜電位vの計算
    # 式(3-13)に相当
    dvdt = np.empty_like(potentials)
    dvdt[set_exc, :] = currents[set_exc, :] / C_EXC
    dvdt[set_inh, :] = currents[set_inh, :] / C_INH

    # すべての微分をひとつの配列にまとめる
    dydt = np.hstack([
        dfdt_ampa, dgdt_ampa,
        dfdt_nmda, dgdt_nmda,
        dfdt_gaba, dgdt_gaba,
        dvdt,
    ])
    return dydt


def simulate_working_memory(t_eval,
                            architecture,
                            dysfuncs_ampa=1.0,
                            dysfuncs_nmda=1.0,
                            dysfuncs_gaba=1.0):
    """ワークングメモリの挙動をシミュレーションする関数

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列

    Returns
    -------
    results : dict
        膜電位やスパイクの変化をリストとして値に保存した辞書
    """
    # [A] シミュレーションの設定
    # シードを固定する
    np.random.seed(SEED)

    # 神経細胞の個数の設定
    num_unit = architecture['num_unit']  # 神経細胞の総数
    num_unit_exc = architecture['num_unit_exc']  # 興奮性神経細胞の総数
    num_unit_inh = architecture['num_unit_inh']  # 抑制性神経細胞の総数

    # [B] 初期値の設定
    f_ampa = np.zeros((num_unit, num_unit_exc))
    g_ampa = np.zeros((num_unit, num_unit_exc))
    f_nmda = np.zeros((num_unit, num_unit_exc))
    g_nmda = np.zeros((num_unit, num_unit_exc))
    f_gaba = np.zeros((num_unit, num_unit_inh))
    g_gaba = np.zeros((num_unit, num_unit_inh))
    potentials = V_INIT * np.ones((num_unit, 1))

    spikes = np.zeros((num_unit, 1))
    last_spikes = -100 * np.ones((num_unit, 1))

    # [C] 結果保存用変数の準備
    results = {
        'potentials': [],
        'spikes': [],
    }

    for t in t_eval:
        # [D] 計算結果を保存
        results['potentials'].append(potentials)
        results['spikes'].append(spikes)

        # [E] 各時刻における計算
        # 変数を一つにまとめる
        y = np.hstack([
            f_nmda, g_nmda,
            f_ampa, g_ampa,
            f_gaba, g_gaba,
            potentials,
        ])

        # ノイズ電流の作成
        currents_noise = np.random.binomial(
            n=1, p=0.1, size=potentials.shape
        ) * CUE_SCALE
        currents_noise_weight = np.where(
            np.array(architecture['excitation_binary']) == 1,
            CUE_SCALE_EXT, CUE_SCALE_INH
        ).reshape(currents_noise.shape)
        currents_noise = 1000 * currents_noise_weight * currents_noise

        # [E-a] 微分方程式による更新
        # コンダクタンスの微分, コンダクタンス, 膜電位が対象
        y = solve_differential_equation(
            dydt=differentiate_working_memory,
            t=t,
            y=y,
            delta_t=DELTA_T,
            method='rk4',
            last_spikes=last_spikes,
            currents_noise=currents_noise,
            dysfuncs_ampa=dysfuncs_ampa,
            dysfuncs_nmda=dysfuncs_nmda,
            dysfuncs_gaba=dysfuncs_gaba,
            architecture=architecture,
        )
        (f_nmda, g_nmda,
         f_ampa, g_ampa,
         f_gaba, g_gaba,
         potentials) = np.split(
            y, architecture['split_list'], axis=1)

        # [E-b] 積分発火モデルによる更新
        refractory = (last_spikes < t) & (t <= last_spikes + T_REF)
        active = (potentials >= V_THERESHOLD) & (~ refractory)
        potentials[active] = V_ACT
        potentials[refractory] = V_RESET

        # [E-c] スパイクの判定
        spikes = np.where(potentials == V_ACT, 1, 0)

        # [E-d] 直近の発火時刻の更新
        last_spikes = np.where(spikes == 1, t, last_spikes)

    # [F] 返り値の準備
    for key, value in results.items():
        # 結果の値をNumpy型に整形
        results[key] = np.hstack(value).T

    return results


# [コラム] ホジキン・ハックスリーモデル
def calc_alpha_m(V):
    alpha_m = (2.5 - 0.1 * (V - E_REST_HH)) / (np.exp(2.5 - 0.1 * (V - E_REST_HH)) - 1.0)
    return alpha_m

def calc_beta_m(V):
    beta_m = 4.0 * np.exp(- (V - E_REST_HH) / 18.0)
    return beta_m

def calc_alpha_h(V):
    alpha_h = 0.07 * np.exp(- (V - E_REST_HH) / 20.0)
    return alpha_h

def calc_beta_h(V):
    beta_h = 1.0 / (np.exp(3.0 - 0.1 * (V - E_REST_HH)) + 1.0)
    return beta_h

def calc_alpha_n(V):
    alpha_n = (0.1 - 0.01 * (V - E_REST_HH)) / (np.exp(1.0 - 0.1 * (V - E_REST_HH)) - 1.0)
    return alpha_n

def calc_beta_n(V):
    beta_n = 0.125 * np.exp(- (V - E_REST_HH) / 80.0)
    return beta_n

def differentiate_hodgkin_huxley(t, y, square=False, **kwargs):
    E_m, m, h, n = y

    C = 1.0
    g_bar_leak = 0.3  # ms/cm^2
    E_leak = 10.6 + E_REST_HH  # mV
    g_bar_na = 120.0  # ms/cm^2
    E_NA = 115.0 + E_REST_HH  # mV
    g_bar_k = 36.0  # ms/cm^2
    E_k = -12.0 + E_REST_HH   # mV

    # 電流の計算
    currents_na = - g_bar_na * np.power(m, 3) * h *(E_m - E_NA)
    currents_k = - g_bar_k * np.power(n, 4) * (E_m - E_k)
    currents_leak = - g_bar_leak * (E_m - E_leak)

    if square:
        # 50ミリ秒ごとに生じる矩形波（オリジナル）
        currents_ext = 0.0 if t % 100 < 50 else 20.0
    else:
        # 山﨑・五十嵐（2021）と同じ設定
        currents_ext = 9.0

    # 微分方程式の計算
    dE_mdt = (1 / C) * (currents_na + currents_k + currents_leak + currents_ext)
    dmdt = calc_alpha_m(E_m) * (1 - m) - calc_beta_m(E_m) * m
    dhdt = calc_alpha_h(E_m) * (1 - h) - calc_beta_h(E_m) * h
    dndt = calc_alpha_n(E_m) * (1 - n) - calc_beta_n(E_m) * n

    dydt = np.hstack([dE_mdt, dmdt, dhdt, dndt])
    return dydt


def simulate_hodgkin_huxley(t_eval,
                            delta_t,
                            method='rk4'):
    """ホジキン・ハックスリーモデルを用いたシミュレーション

    Parameters
    ----------
    t_eval : np.ndarray or list
        評価の対象となる時刻を保存した一次元の配列
    delta_t : float
        ステップ幅 (ミリ秒)
    method : str
        'euler'の場合オイラー法，'rk4'の場合ルンゲ・クッタ法を使用

    Returns
    -------

    Note
    ----
    神経細胞の数は一つのみ対応

    時間幅（delta_t）の値について
    - Eulerの場合: ステップ幅0.1は厳しい。0.01は必要か
    - RK4の場合: ステップ幅0.1は厳しい。0.01は必要か

    定数値について
    - 山﨑・五十嵐（2021）の数字を使っている
    - なお，山﨑・五十嵐（2021）では，delta_t = 10μs = 0.01msでルンゲクッタ
    - また「HHモデルをオイラー法で解こうとすると1μs程度は必要と」記載あり
    """

    # [B] 初期値の設定
    y = np.hstack([
        E_REST_HH,
        calc_alpha_m(E_REST_HH) / (calc_alpha_m(E_REST_HH) + calc_beta_m(E_REST_HH)),
        calc_alpha_h(E_REST_HH) / (calc_alpha_h(E_REST_HH) + calc_beta_h(E_REST_HH)),
        calc_alpha_n(E_REST_HH) / (calc_alpha_n(E_REST_HH) + calc_beta_n(E_REST_HH)),
    ])

    # [C] 結果保存用変数の準備
    results = {
        'E': [],
        'm': [],
        'h': [],
        'n': [],
    }

    for step, t in enumerate(t_eval):
        # [D] 計算結果を保存
        results['E'].append(y[0])
        results['m'].append(y[1])
        results['h'].append(y[2])
        results['n'].append(y[3])

        # [E] 各時刻における計算
        y = solve_differential_equation(
            dydt=differentiate_hodgkin_huxley,
            t=t,
            y=y,
            delta_t=delta_t,
            method=method,
        )
    return results


# 補足：微分方程式ソルバーのデバッグ関数
def differentiate_exponential(t, y, **kwargs):
    if 'tau' in kwargs.keys():
        tau = kwargs['tau']
    else:
        tau = 1.0
    dydt = - y / tau
    return dydt


def simulate_exponential(t_eval,
                         delta_t,
                         method='euler',
                         tau=1.0,
                         y_init=10.0,
                         proceedure='original',
                         ):
    y_init = 10.0
    y = y_init
    results = {
        'y': [],
        't_eval': t_eval,
    }

    if proceedure == 'original':
        for t in t_eval:
            # [D] 計算結果を保存
            results['y'].append(y)

            # [E] 各時刻における計算
            # [E-a] 微分方程式による更新
            y = solve_differential_equation(
                dydt=differentiate_exponential,
                t=t,
                y=y,
                delta_t=delta_t,
                method=method,
                tau=tau,
            )
        results['y'] = np.array(results['y']).reshape(-1, 1)

    elif proceedure == 'exact':
        # results['y'] = np.exp(- t_eval / tau + np.log(y_init)).reshape(-1, 1)
        results['y'] = y_init * np.exp(- t_eval / tau).reshape(-1, 1)

    elif proceedure == 'scipy':
        method = 'RK45' if method == 'rk4' else method
        rslt = solve_ivp(
            fun=differentiate_exponential,
            t_span=[t_eval[0], t_eval[-1]],
            y0=[y_init],
            method=method,
            t_eval=t_eval,
            tau=tau,
        )
        results['y'] = rslt.y.T
        results['all'] = rslt

    else:
        NotImplementedError()

    return results


def differentiate_circle(t, y, **kwargs):
    u, v = y[0, 0], y[1, 0]
    dydt = np.array([[-v, u]]).T
    return dydt


def simulate_circle(t_eval,
                    delta_t,
                    method='euler',
                    proceedure='original'):

    # [B] 初期値の設定
    y = np.array([[1.0, 0.0]]).T

    # [C] 結果保存用変数の準備
    results = {
        'y': [],
        't_eval': t_eval,
    }

    if proceedure == 'original':
        for t in t_eval:
            # [D] 計算結果を保存
            results['y'].append(y)

            # [E] 各時刻における計算
            y = solve_differential_equation(
                dydt=differentiate_circle,
                t=t,
                y=y,
                delta_t=delta_t,
                method=method,
            )
        results['y'] = np.hstack(results['y']).T

    elif proceedure == 'exact':
        results['y'] = np.stack(
            [np.cos(t_eval),
             np.sin(t_eval)],
            axis=1
        )

    elif proceedure == 'scipy':
        NotImplementedError()
        # method = 'RK45' if method == 'rk4' else method
        # rslt = solve_ivp(
        #     fun=differentiate_circle,
        #     t_span=[t_eval[0], t_eval[-1]],
        #     y0=y[:, 0],
        #     method=method,
        #     t_eval=t_eval,
        # )
        # results['y'] = rslt.y.T
        # results['all'] = rslt

    else:
        NotImplementedError()

    return results


def differentiate_oscillation_damped(t, y, **kwargs):
    """減衰振動の微分方程式

    Notes
    -----
    xは座標（位置），vは速度
    kはバネ定数，cは減衰係数，mは質量

    減衰振動の厳密計算は以下を参考
    https://watlab-blog.com/2019/06/10/python-1dof-mck/
    """
    k, c, m = kwargs['k'], kwargs['c'], kwargs['m']
    x, v = np.split(y, [1], axis=0)
    dxdt = v
    dvdt = (-k*x - c*v) / m
    dydt = np.vstack([dxdt, dvdt])
    # print('diff', y, x, v, dxdt, dvdt, dydt)
    return dydt


def simulate_oscillation_damped(t_eval,
                                delta_t,
                                method='euler',
                                proceedure='original'
                                ):
    """
    Parameters
    ----------
    t_eval :
    method : str
        近似解以外にも`method='exact'`を指定することで厳密解を計算可能
    delta_t :

    Returns
    -------
    resutls : dict

    Notes
    -----
    減衰振動の厳密計算は以下を参考
    https://watlab-blog.com/2019/06/10/python-1dof-mck/
    """
    # [A] シミュレーションの設定
    k, c, m = 0.2, 0.1, 1.0

    # [B] 初期値の設定
    # y = np.zeros((2, 1))
    y = np.array([[0.1], [0.0]])

    # [C] 結果保存用変数の準備
    results = {
        'y': [],
        't_eval': t_eval,
    }

    if proceedure == 'original':
        for t in t_eval:
            # [D] 計算結果を保存
            results['y'].append(y)

            # [E] 各時刻における計算
            y = solve_differential_equation(
                dydt=differentiate_oscillation_damped,
                t=t,
                y=y,
                delta_t=delta_t,
                method=method,
                k=k,
                c=c,
                m=m,
            )
        results['y'] = np.hstack(results['y']).T

    elif proceedure == 'exact':
        state0 = y[0, 0], y[1, 0]

        zeta = c / (2 * np.sqrt(m * k))
        omega = np.sqrt(k / m)
        omega_d = omega * np.sqrt(1 - np.power(zeta, 2))
        sigma = omega_d * zeta
        X = np.sqrt(np.power(state0[0], 2) \
                    + np.power((state0[1] + sigma * state0[0])/omega_d, 2))
        phi = np.arctan((state0[1] + (sigma * state0[0]))/(state0[0] * omega_d))

        theory = np.exp(- sigma * t_eval) * X * np.cos(omega_d * t_eval - phi)
        results['y'] = theory.reshape(-1, 1)

    elif proceedure == 'scipy':
        NotImplementedError()
    else:
        NotImplementedError()

    return results


def compare_solver(equation_type,
                   t_max=20,
                   method_list=['euler', 'rk4'],
                   delta_t_list=[1.0, 0.1, 0.001],
                   execute_scipy=False,
                   ):
    """微分方程式ソルバーのデバッグ関数

    Parameters
    ----------
    equation_type : str
        使用する微分方程式の種類
        現在実装されているのは以下の値
        'exponential' : 指数関数
        'circle' : 円軌道
        'oscillation_dampled' : 減衰振動
    t_max : float
    method_list : list of str
    delta_t_list : list of float
    execute_scipy : bool
        SciPy Solverでのシミュレーションを実行するかどうか
        `simulate_exponential()`のみ対応可能

    Returns
    -------
        異なる時間幅，計算法でシミュレーションした場合の結果
    """
    # 使用する微分方程式の選択
    funcs_simulate = globals().get('simulate_' + equation_type)
    if funcs_simulate is None:
        raise Exception(equation_type + ' is not implemented.')
    else:
        print(funcs_simulate, 'was selected...')

    # 結果保存用変数の準備
    results_all = {
        'settings': {
            'method': method_list,
            'delta_t': delta_t_list,
        },
        'exact': dict([
            (str(delta_t), None) for delta_t in delta_t_list]),
        'original': dict([
            (method, dict([
                (str(delta_t), None) for delta_t in delta_t_list]))
            for method in method_list]),
    }
    if execute_scipy:
        results_all['scipy'] = dict([
            (method, dict([
                (str(delta_t), None) for delta_t in delta_t_list]))
            for method in method_list])
    print("Save Dictionary: ", results_all)

    # 厳密解
    for delta_t in delta_t_list:
        t_eval = generate_t_eval(
            t_max=t_max,
            delta_t=delta_t,
        )
        results = funcs_simulate(
            t_eval=t_eval,
            delta_t=delta_t,
            proceedure='exact',
        )
        results_all['exact'][str(delta_t)] = results

    # 自作関数
    for delta_t, method in itertools.product(delta_t_list, method_list):
        print(
            'Now, simulation is executed using original solver...',
            '(', method, delta_t, ')',
        )
        t_eval = generate_t_eval(
            t_max=t_max,
            delta_t=delta_t,
        )
        results = funcs_simulate(
            t_eval=t_eval,
            delta_t=delta_t,
            method=method,
            proceedure='original',
        )
        results_all['original'][method][str(delta_t)] = results

    # SciPy Solver
    if execute_scipy:
        for delta_t, method in itertools.product(delta_t_list, method_list):
            print(
                'Now, simulation is executed using SciPy solver...',
                '(', method, delta_t, ')',
            )
            t_eval = generate_t_eval(
                t_max=t_max,
                delta_t=delta_t,
            )
            results = funcs_simulate(
                t_eval=t_eval,
                delta_t=delta_t,
                method=method,
                proceedure='scipy',
            )
            results_all['scipy'][method][str(delta_t)] = results

    return results_all


def validate_solver():
    """ライブラリによる数値計算のズレをチェックする

    自作数値計算関数の検証用

    Parameters
    ----------

    Returns
    -------

    """
    equation_type_list = [
        "exponential",
        "circle",
        "oscillation_damped",
    ]
    t_max = 20
    delta_t_list=[
        1.0,
        0.01,
        0.0001,
    ]
    results = {}

    for equation_type in equation_type_list:
        print(
            "\nSimulation of", equation_type,
            "\n============================="
        )
        # 設定値
        if equation_type in ["exponential"]:
            method_list = ['rk4']
            proceedure_list = ["original", "exact", "scipy"]
            execute_scipy = True
        elif equation_type in ["circle", "oscillation_damped"]:
            method_list = ['euler', 'rk4']
            proceedure_list = ["original", "exact"]
            execute_scipy = False
        else:
            raise Exception(equation_type + ' is not implemented.')

        # 数値計算
        rslt = compare_solver(
            equation_type=equation_type,
            t_max=t_max,
            method_list=method_list,
            delta_t_list=delta_t_list,
            execute_scipy=execute_scipy,
        )

        # 可視化
        print('Now, plotting results...')
        for delta_t in delta_t_list:
            for method in method_list:
                plt.plot(
                    rslt["original"][method][str(delta_t)]['t_eval'],
                    rslt["original"][method][str(delta_t)]['y'][:, 0],
                    linewidth=0.5,
                    alpha=0.8,
                    label="Original / " + method,
                )
                if execute_scipy:
                    plt.plot(
                        rslt["scipy"][method][str(delta_t)]['t_eval'],
                        rslt["scipy"][method][str(delta_t)]['y'][:, 0],
                        linewidth=0.5,
                        alpha=0.8,
                        label="SciPy / " + method,
                    )
            plt.plot(
                rslt["exact"][str(delta_t)]['t_eval'],
                rslt["exact"][str(delta_t)]['y'][:, 0],
                linewidth=0.5,
                alpha=0.8,
                label="Exact",
            )

            plt.title(equation_type + " / " + str(delta_t))
            plt.legend()
            plt.show()
            plt.close()

        # 平均二乗誤差
        for delta_t in delta_t_list:
            y_exact = rslt["exact"][str(delta_t)]['y']

            for method in method_list:
                mse_exact = np.mean(np.square(
                    rslt["original"][method][str(delta_t)]['y'] \
                    - y_exact
                ))
                print(
                    "MSE between my library and exact solution",
                    "(", method, delta_t, "):",
                     mse_exact,
                )

                if execute_scipy:
                    mse_scipy = np.mean(np.square(
                        rslt["original"][method][str(delta_t)]['y'] \
                        - rslt["scipy"][method][str(delta_t)]['y']
                    ))
                    print(
                        "MSE between my library and SciPy solver",
                        "(", method, delta_t, "):",
                        mse_scipy
                    )
                    mse_scipy = np.mean(np.square(
                        rslt["original"][method][str(delta_t)]['y'] \
                        - y_exact
                    ))
                    print(
                        "MSE between SciPy solver and exact solution",
                        "(", method, delta_t, "):",
                        mse_scipy
                    )

        # 結果を保存
        results[equation_type] = rslt

    return results
