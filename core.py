# core.py – 2009 Tetris Guideline 準拠・共有ロジック (Row/Column Bitboard + Incremental)
from __future__ import annotations
import random
from typing import List, Tuple, Optional, Dict
import heapq

# ────────── 定数 ──────────
MATRIX_W, MATRIX_H = 10, 20
FULL_ROW          = (1 << MATRIX_W) - 1          # 0b1111111111
W, H              = MATRIX_W, MATRIX_H           # エイリアス
COLORS = {
    'I': (0,255,255), 'O': (255,255,0), 'T': (128,0,128),
    'S': (0,255,0),   'Z': (255,0,0),   'J': (0,0,255),
    'L': (255,165,0), 'G': (90,90,90),
}
PIECE_SHAPES: Dict[str, List[List[Tuple[int,int]]]] ={
    'I': [[(-1,0),(0,0),(1,0),(2,0)], [(1,-1),(1,0),(1,1),(1,2)],
          [(-1,1),(0,1),(1,1),(2,1)], [(0,-1),(0,0),(0,1),(0,2)]],
    'O': [[(0,0),(1,0),(0,1),(1,1)]]*4,
    'T': [[(-1,0),(0,0),(1,0),(0,1)], [(0,-1),(0,0),(1,0),(0,1)],
          [(-1,0),(0,0),(1,0),(0,-1)],[(0,-1),(0,0),(-1,0),(0,1)]],
    'S': [[(-1,0),(0,0),(0,1),(1,1)], [(0,1),(0,0),(1,0),(1,-1)],
          [(1,0),(0,0),(0,-1),(-1,-1)],[(0,-1),(0,0),(-1,0),(-1,1)]],
    'Z': [[(0,0),(1,0),(-1,1),(0,1)],  [(0,0),(0,-1),(1,1),(1,0)],
          [(0,0),(-1,0),(1,-1),(0,-1)],[(0,0),(0,1),(-1,-1),(-1,0)]],
    'J': [[(-1,0),(0,0),(1,0),(-1,1)], [(0,1),(0,0),(0,-1),(1,1)],
          [(1,0),(0,0),(-1,0),(1,-1)], [(0,-1),(0,0),(0,1),(-1,-1)]],
    'L': [[(-1,0),(0,0),(1,0),(1,1)],  [(0,1),(0,0),(0,-1),(1,-1)],
          [(1,0),(0,0),(-1,0),(-1,-1)],[(0,-1),(0,0),(0,1),(-1,1)]],
}

JLSTZ_KICKS = {(0,1):[(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
               (1,0):[(0,0),(1,0),(1,-1),(0,2),(1,2)],
               (1,2):[(0,0),(1,0),(1,-1),(0,2),(1,2)],
               (2,1):[(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)],
               (2,3):[(0,0),(1,0),(1,1),(0,-2),(1,-2)],
               (3,2):[(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],
               (3,0):[(0,0),(-1,0),(-1,-1),(0,2),(-1,2)],
               (0,3):[(0,0),(1,0),(1,1),(0,-2),(1,-2)]}
I_KICKS = {(0,1):[(0,0),(-2,0),(1,0),(-2,-1),(1,2)],
           (1,0):[(0,0),(2,0),(-1,0),(2,1),(-1,-2)],
           (1,2):[(0,0),(-1,0),(2,0),(-1,2),(2,-1)],
           (2,1):[(0,0),(1,0),(-2,0),(1,-2),(-2,1)],
           (2,3):[(0,0),(2,0),(-1,0),(2,1),(-1,-2)],
           (3,2):[(0,0),(-2,0),(1,0),(-2,-1),(1,2)],
           (3,0):[(0,0),(1,0),(-2,0),(1,-2),(-2,1)],
           (0,3):[(0,0),(-1,0),(2,0),(-1,2),(2,-1)]}


# ───────────── グリッド Piece / Board (GUI 用) ─────────────
class Piece:
    def __init__(self, kind:str):                   # 出現位置はガイドライン準拠
        self.kind, self.rot, self.x, self.y = kind, 0, 4, 21
    def cells(self, dx=0, dy=0):
        for bx,by in PIECE_SHAPES[self.kind][self.rot]:
            yield self.x+bx+dx, self.y+by+dy
    def rotate(self, d: int, board: "Board") -> bool:
        """SRS キック込み回転 (Clockwise:+1, CCW:-1)"""
        old, new = self.rot, (self.rot + d) % 4
        kicks = I_KICKS if self.kind == 'I' else JLSTZ_KICKS
        for kx, ky in kicks.get((old, new), [(0, 0)]):
            if board.valid(self, new, kx, ky):
                self.rot, self.x, self.y = new, self.x + kx, self.y + ky
                return True
        return False

class Board:
    def __init__(self):
        self.grid = [[0]*MATRIX_W for _ in range(MATRIX_H)]
    # ── 判定
    def valid(self, p:Piece, r:int, dx=0, dy=0):
        for bx,by in PIECE_SHAPES[p.kind][r]:
            x, y = p.x+bx+dx, p.y+by+dy
            if x<0 or x>=MATRIX_W or y<0: return False
            if y<MATRIX_H and self.grid[y][x]: return False
        return True
    # ── 固定 / ライン消去
    def lock(self, p:Piece):
        for x,y in p.cells():
            if 0<=y<MATRIX_H: self.grid[y][x] = p.kind
    def clear(self) -> int:
        """揃った行を消して上から空行を補充（grid[0] が最下段想定）"""
        self.grid = [r for r in self.grid if not all(r)]   # 1) そろった行を除去
        cleared   = MATRIX_H - len(self.grid)              # 2) 消えた行数
        while len(self.grid) < MATRIX_H:                   # 3) 上端に空行を追加
            self.grid.append([0] * MATRIX_W)               #   append が “上”
        return cleared

# ────────── Row/Column Bitboard Board State ──────────
class BoardState:
    """列 20bit ×10 で保持。行クリアも O(列) 。"""
    __slots__=("cols","h","holes","cov","rough","bump","_hash","_dirty",
               "well_depth", "well_cells",
                 "t_spot", "parity",
                 "_hash", "_dirty")
    def __init__(self, cols:Tuple[int,...]|None=None):
        self.cols = cols or tuple(0 for _ in range(W))
        self._dirty=True      # 次回 features で再計算
        # 変数の入れ物
        self.h=[0]*W; self.holes=self.cov=0; self.rough=self.bump=0; self._hash=0
        self._recalc()

    # ─ clone / hash ─
    def clone(self): return BoardState(self.cols)
    def __hash__(self): return self._hash
    def __eq__(self,o): return self.cols==o.cols

    # ─ cell helpers ─
    def cell(self,x:int,y:int)->bool: return ((self.cols[x]>>y)&1)!=0
    # ── 1 マス立てる（外部からも使う） ──────────────────
    def set_cell(self, x:int, y:int):
        """GUI/DAG が呼び出す公開 API。内部では _raw_set() に委譲。"""
        self._raw_set(x, y)

    # 内部実装を分離（clone 内ループからも呼ぶ）
    def _raw_set(self, x:int, y:int):
        col = list(self.cols)
        col[x] |= 1 << y
        self.cols = tuple(col)
        self._dirty = True


    # 旧 clear_lines() との後方互換ラッパ
    def clear_lines(self) -> int:
        """全 20 行スキャン（旧コード互換）"""
        return self._clear_rows(set(range(H)))

    # DAG 用・**置いた行だけ** を渡せる高速版
    def clear_rows_fast(self, touched: set[int]) -> int:
        """touched 行だけ判定して消去"""
        return self._clear_rows(touched)
    
    # ─ line clear ─
    def _clear_rows(self, touched_rows:set[int])->int:
        """touched_rows に 1bit でも置いた行だけをチェックして高速化"""
        fulls=[r for r in touched_rows if all((c>>r)&1 for c in self.cols)]
        if not fulls: return 0
        cleared=len(fulls)
        new=list(self.cols)
        for i,c in enumerate(new):
            for r in fulls:
                lower=c & ((1<<r)-1)
                upper=c>>(r+1)
                c=lower|(upper<<r)
            new[i]=c
        self.cols=tuple(new); self._dirty=True
        return cleared

    # ─ hard-drop ─
    def drop_piece(self,k:str,r:int,x:int)->Optional[Tuple['BoardState',int]]:
        shape=PIECE_SHAPES[k][r]
        y=max(self.h[x+dx]-dy for dx,dy in shape)
        top = y+max(dy for _,dy in shape)
        if top>=H: return None      # top-out
        new=self.clone()
        touched=set()
        for dx,dy in shape:
            new._raw_set(x + dx, y + dy)
            touched.add(y+dy)
        lines=new._clear_rows(touched)
        return new,lines

    # ─ recalc 全特徴 ─
    def _recalc(self):
        # 高さ
        self.h=[c.bit_length() for c in self.cols]
        # 穴 & covered
        holes=cov=0
        for x,col in enumerate(self.cols):
            h=self.h[x]
            filled=col & ((1<<h)-1)
            holes += h - filled.bit_count()
            mask=1
            for y in range(h):
                if not (filled&mask) and (filled & (mask-1)): cov+=1
                mask<<=1
        # 粗さ & bump
        deltas=[abs(self.h[i]-self.h[i+1]) for i in range(W-1)]
        maxd=max(deltas) if deltas else 0
        bump=sum(-(d if d<=3 else d*d) for d in deltas if d!=maxd)
        hbar=sum(self.h)/W
        rough=sum(abs(h-hbar) for h in self.h)

        self.holes,self.cov,self.bump,self.rough=holes,cov,bump,rough
        
        
        # 3-A) 井戸 (左右より低いセル数)
        well_depth = well_cells = 0
        for x in range(W):
            l = self.h[x-1] if x > 0 else H
            r = self.h[x+1] if x < W-1 else H
            min_lr = min(l, r)
            depth = max(0, min_lr - self.h[x])
            well_depth += depth
            well_cells += depth * (depth + 1) // 2    # 1+2+…depth

        # 3-B) T-slot 検出 (中心 4 マスが空で対角 3/4 埋まっている形)
        t_spot = 0
        for y in range(1, H):          # 下 1 行は無視
            row_bits = [(self.cols[x] >> y) & 1 for x in range(W)]
            above    = [(self.cols[x] >> (y+1)) & 1 for x in range(W)]
            for x in range(1, W-1):
                if row_bits[x] == 0 and above[x] == 0:
                    # 周囲斜め 4 マスのうち 3 以上が埋まっていれば T-slot
                    diag = (
                        row_bits[x-1] + row_bits[x+1] +
                        above[x-1]    + above[x+1]
                    )
                    if diag >= 3:
                        t_spot += 1

        # 3-C) パリティ (行ごとの 1bit xor)
        parity = 0
        for y in range(H):
            bits = sum((c >> y) & 1 for c in self.cols) & 1
            parity ^= bits

        # 3-D) 保存
        self.well_depth = well_depth
        self.well_cells = well_cells
        self.t_spot     = t_spot
        self.parity     = parity

        # 既存変数 holes…rough も計算済み
        self._hash = hash(self.cols)
    
        self._hash=hash(self.cols); self._dirty=False

    # ─ public: drop (GUI 用互換) ─
    def drop(self,k,r,x):                 # GUI / GA 互換
        res=self.drop_piece(k,r,x)
        if res is None: raise ValueError('top_out')
        return res

    def features(self)->Dict[str,float]:
        if self._dirty: self._recalc()
        return {
            'holes':self.holes,
            'covered_holes':self.cov,
            'roughness':self.rough,
            'bump_pen':self.bump,
            'height':sum(self.h),
            'max_h':max(self.h),
            # row/col_trans は速度優先で除外（使わないなら GA 側で重み 0に）
            'row_trans':0,'col_trans':0,
            'well_depth': self.well_depth,
            'well_cells': self.well_cells,
            't_spot': self.t_spot,
            'parity': self.parity,
        }

    # ─ helper: Board → BitBoard ─
    @staticmethod
    def from_board(b:'Board|BoardState'):
        if isinstance(b,BoardState): return b.clone()
        cols=[0]*W
        for y,row in enumerate(b.grid):
            for x,v in enumerate(row):
                if v: cols[x]|=1<<y
        return BoardState(tuple(cols))
    @staticmethod
    def empty() -> "BoardState":
        """空盤面を返す（GA / テスト用）"""
        return BoardState()                 # cols=None → 全 0 で初期化

# ──────────────────  Low-level helpers  ──────────────────
def _valid_pos(st: BoardState, kind: str, rot: int, x: int, y: int) -> bool:
    """盤外 - or 既存ブロック衝突があれば False。
       y は可視領域(0-19)より上でも構わない。"""
    for dx, dy in PIECE_SHAPES[kind][rot]:
        xx, yy = x + dx, y + dy
        if xx < 0 or xx >= MATRIX_W:      # 左右壁
            return False
        if yy < 0:                        # 床より下
            return False
        if yy < MATRIX_H and st.cell(xx, yy):     # 既存ブロック
            return False
    return True


def _kick_tests(kind: str, rot_from: int, rot_to: int):
    """SRS キックテーブルを返却"""
    if kind == 'I':
        return I_KICKS.get((rot_from, rot_to), [(0, 0)])
    else:
        return JLSTZ_KICKS.get((rot_from, rot_to), [(0, 0)])



# ───────────── 全手列挙 (ビット盤面) ─────────────
_MOVE_CACHE:Dict[Tuple[int,str],List[Tuple[int,int,int,BoardState]]]={}
# ────────────────────  DAG 着手列挙  ────────────────────
def all_moves_dag(st: BoardState, kind: str) -> list[tuple[int, int, int, BoardState]]:
    """
    Cold Clear と同様:
      1. 生成位置 (x=4, y=21 付近) から BFS。
      2. <←,→,CW,CCW,↓> で辿れる “到達可能 (rot,x,y)” を列挙。
      3. そこから 1cell 落ちられない位置を『着地候補』とし，
         実際に固定して BoardState と消去行数を作る。
    戻り値は [(rot, x, lines, next_state), …]
    """
    # ─ 生成座標 (ガイドライン準拠: y=21 は Matrix 上 1 行上)
    spawn_x = 4
    spawn_y = MATRIX_H + 1        # 21 (=20+1)

    # 初期回転は 0 だけ (SRS)
    start = (0, spawn_x, spawn_y)
    if not _valid_pos(st, kind, *start):
        return []                 # そもそも湧けない＝TOP OUT
    
    
    cache_key = (hash(st), kind)
    if cache_key in _MOVE_CACHE:
        return _MOVE_CACHE[cache_key]
    
    # BFS
    visited: set[tuple[int, int, int]] = set()
    q = [start]
    landings: list[tuple[int, int, int, BoardState]] = []

    while q:
        rot, x, y = q.pop()
        if (rot, x, y) in visited:
            continue
        visited.add((rot, x, y))

        # ↓ が無理なら “着地”
        if not _valid_pos(st, kind, rot, x, y - 1):
            # Hard-drop→固定して盤面生成
            next_st = st.clone()
            for dx, dy in PIECE_SHAPES[kind][rot]:
                next_st.set_cell(x + dx, y + dy)
            touched = {y+dy for _, dy in PIECE_SHAPES[kind][rot]}   # 置いた行だけ
            lines   = next_st.clear_rows_fast(touched)
            landings.append((rot, x, lines, next_st))
            
            # ↓ もう動けないので neighbours 生成は不要
            continue

        # ── neighbours ───────────────────
        # (1) 左右移動
        if _valid_pos(st, kind, rot, x - 1, y):
            q.append((rot, x - 1, y))
        if _valid_pos(st, kind, rot, x + 1, y):
            q.append((rot, x + 1, y))

        # (2) 回転 ±1  (SRSキック適用)
        for drot in (1, -1):
            rot2 = (rot + drot) & 3
            for kx, ky in _kick_tests(kind, rot, rot2):
                nx, ny = x + kx, y + ky
                if _valid_pos(st, kind, rot2, nx, ny):
                    q.append((rot2, nx, ny))
                    break   # 1 つ目が通れば十分

        # (3) ソフトドロップ１段
        q.append((rot, x, y - 1))
        
    _MOVE_CACHE[cache_key] = landings
    return landings
# ───────────── Heuristic AI ─────────────
FEATURES=[
    'single','double','triple','tetris','b2b',
    'row_trans','col_trans','roughness',
    'covered_holes','bump_pen','holes',
    'height','max_h',
        'well_depth','well_cells','t_spot','parity',
]
DEFAULT_W={
    'single':-0.2,'double':0.05,'triple':1.5,'tetris':10.0,'b2b':3.0,
    'row_trans':-0.2,'col_trans':-0.2,'roughness':-0.1,
    'covered_holes':-4.8,'bump_pen':1.5,'holes':-0.4,
    'height':-0.3,'max_h':-0.2,
    'well_depth' :  0.45,   # 井戸は Good → 正
    'well_cells' : -0.05,   # 深過ぎペナルティ
    't_spot'     :  2.00,   # T-Spin は高ボーナス
    'parity'     : -0.2,    # 偶奇ズレ△
}

# ────────── ヒューリスティック AI ──────────
class HeuristicAI:
    def __init__(self,w:Dict[str,float]=None,depth:int=5,beam:int=15,*,use_bb=True):
        self.w=w or DEFAULT_W; self.depth=depth; self.beam=beam
    def _s(self,f): return sum(self.w.get(k,0)*v for k,v in f.items())

    def best_move(self,board:'Board|BoardState',kind0:str,nexts:List[str]|None=None):
        root=BoardState.from_board(board)
        frontier=[(0.0,root,[],False)]     # (score, state, seq, last_major)
        kinds=[kind0]+(nexts or [])[:self.depth-1]
        for d,k in enumerate(kinds):
            width=max(1,int(self.beam*0.8**d))
            cand=[]
            for sc,st,seq,last in frontier:
                for r,x,ln,st2 in all_moves_dag(st,k):
                    feat=st2.features()
                    feat.update({'single':ln==1,'double':ln==2,'triple':ln==3,'tetris':ln==4})
                    major=(ln==4)
                    feat['b2b']=1 if last and major else 0
                    cand.append((sc+self._s(feat),st2,seq+[(r,x)],major))
            if not cand: return None
            frontier=heapq.nlargest(width,cand,key=lambda t:t[0])
        return frontier[0][2][0]
    
    
    
   