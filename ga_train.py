# ga_train.py – DEAP + multiprocessing で GA 最適化（前回ベストを初期個体に注入）

from __future__ import annotations
import random, json, pathlib, pickle, multiprocessing as mp
import numpy as np
from deap import base, creator, tools
from core import BoardState, HeuristicAI, FEATURES, DEFAULT_W           # ← core.py から共通定義

import time

# ───────────────────────── パラメータ ─────────────────────────
POP, GEN = 8, 8                # 個体数と世代数を増やす
MAX_PIECES = 3000                # 最大ピース数を増やす
LINE_TARGET = 40                 # 目標ライン数は維持
TOP_PENALTY = 2_000              # ペナルティを増加

# ───────────────────────── 評価用シミュレータ ────────────────
def run_game(weights, seed=0):
    rnd  = random.Random(seed)

    # ── 7-bag を 2 本先まで用意 ───────────────────
    def new_bag():
        b = list('IOTSZJL'); rnd.shuffle(b); return b
    bag         = new_bag()      # 今使っているバッグ
    future_bag  = new_bag()      # 次のバッグ

    st   = BoardState.empty()
    # ビットボード版コアで評価
    ai   = HeuristicAI(dict(zip(FEATURES, weights)), depth=5)

    lines_total = 0
    pieces      = 0

    while pieces < MAX_PIECES and lines_total < LINE_TARGET:

        # ─── 次ミノを取り出し ─────────────────────
        if not bag:                      # 今の bag が空になったら
            bag, future_bag = future_bag, new_bag()

        kind = bag.pop()                 # 現在ミノ

        # ─── Next4 を組み立てる (bag[::-1] で先頭→末尾順) ───
        next4 = (bag[::-1] + future_bag[::-1])[:4]

        # ─── 最善手を 5-ply で探索 ─────────────────
        move = ai.best_move(st, kind, next4)
        if move is None:                     # 置ける場所が無い → トップアウト
            break
        rot, x = move

        # ─── ドロップ ───────────────────────────
        try:
            st, cleared = st.drop(kind, rot, x)
        except ValueError:                  # 衝突 / top_out
            break

        lines_total += cleared
        pieces      += 1

    # テトリスボーナスを追加
    tetris_count = 0
    for _ in range(pieces):
        if cleared == 4:
            tetris_count += 1
    
    # スコア計算を修正
    score = -pieces + (tetris_count * 10)  # テトリスごとにボーナス
    
    if lines_total < LINE_TARGET:
        score -= TOP_PENALTY
    
    return score


# ───────────────────────── DEAP 準備 ─────────────────────────
creator.create('FitMax', base.Fitness, weights=(-1.0,))
creator.create('Ind', list, fitness=creator.FitMax)

tb = base.Toolbox()
tb.register('attr', random.uniform, -1, 1)
tb.register('individual',
            tools.initRepeat, creator.Ind,
            tb.attr, n=len(FEATURES))   # ← 自動で新しい長さになる
tb.register('population',  tools.initRepeat, list, tb.individual)

tb.register('mate',   tools.cxBlend,     alpha=0.5)
tb.register('mutate', tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
tb.register('select', tools.selTournament, tournsize=3)

def evaluate(ind):
    # 3 つの乱数シードで平均スコア
    return (np.mean([run_game(ind, s) for s in (0, 1, 2)]),)
tb.register('evaluate', evaluate)

# ───────────────────────── main ──────────────────────────────
def main():

    # 1) ─── 初期個体を生成 ─────────────────────
    pop = tb.population(POP)

    # 1-a) 前回ベストがあれば個体 0 に注入
    best_path = pathlib.Path('best_weights.json')
    if best_path.exists():
        try:
            data = json.loads(best_path.read_text())
            if all(k in data for k in FEATURES):
                seed_ind = creator.Ind([data[f] for f in FEATURES])
                pop[0]   = seed_ind                       # 0 番目を置き換え
                print('Seed individual loaded from best_weights.json')
        except Exception as e:
            print('Warning: best_weights.json load error ->', e)
    # さらに、必ずデフォルト重みも個体として加える例
    default_ind = creator.Ind([DEFAULT_W[f] for f in FEATURES])
    pop[1 % POP] = default_ind
    print('Default-weight individual injected into generation 0')
    
    # 2) ─── 進化ループ ───────────────────────
    ELITE_SIZE = 2   # 毎世代キープする上位何個体か
    with mp.Pool() as pool:
        for g in range(GEN):
            # ── 評価 ─────────────────────────
            fits = pool.map(tb.evaluate, pop)
            for ind, fit in zip(pop, fits):
                ind.fitness.values = fit

            # ── ロギング ─────────────────────
            best_now = max(pop, key=lambda i: i.fitness.values[0])
            avg_now  = np.mean([i.fitness.values[0] for i in pop])
            print(f'Gen {g:02d}: best {-best_now.fitness.values[0]:.1f}, '
                  f'avg {-avg_now:.1f}  '
                  f'w={[round(w, 3) for w in best_now]}')

            # ── エリート保存 ───────────────────
            elites = tools.selBest(pop, ELITE_SIZE)

            # ── 残りをトーナメント選択 ────────────
            selected = tb.select(pop, len(pop) - ELITE_SIZE)
            children = list(map(tb.clone, selected))

            # ── 交叉 ─────────────────────────
            for c1, c2 in zip(children[::2], children[1::2]):
                if random.random() < 0.7:
                    tb.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            # ── 突然変異 ───────────────────────
            for c in children:
                if random.random() < 0.2:
                    tb.mutate(c)
                    del c.fitness.values

            # ── 次世代生成 ─────────────────────
            pop = elites + children
        # ── ループ後：最終世代を評価 ─────────
        fits = pool.map(tb.evaluate, pop)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

    # 3) ─── ベスト個体保存 ───────────────────
    best = max(pop, key=lambda i: i.fitness.values[0])
    weights_dict = {f: w for f, w in zip(FEATURES, best)}
    pathlib.Path('best_weights.json').write_text(json.dumps(weights_dict))
    print('Saved → best_weights.json')


if __name__ == '__main__':
    mp.freeze_support()          # Windows 用
    start_all = time.time()
    main()
    end_all = time.time()
    print(f"Total GA training time: {end_all - start_all:.1f} sec")