# gui.py – pygame Tetris (Next-5 表示)
from __future__ import annotations
import random, json, pathlib, pygame as pg, numpy as np

from core import (MATRIX_W, MATRIX_H, COLORS, PIECE_SHAPES,
                  Piece, Board, HeuristicAI, DEFAULT_W)


BLOCK, FPS = 30, 60
LOCK_DELAY_MS, MOVE_DELAY_MS, MOVE_REPEAT_MS = 500, 190, 16
GRAVITY = [48,43,38,33,28,23,18,13,8,6,5,5,5,4,4,4,3,3,3,2]

# ── GA 重み読込 ────────────────────────────────
p=pathlib.Path('best_weights.json')
weights=json.loads(p.read_text()) if p.exists() else DEFAULT_W
AI=HeuristicAI(weights, use_bb=True)



# ───────────────── Game ─────────────────────────
class Game:
    def __init__(self, ai_mode=True):
        self.board=Board()
        self.bag=[]
        self.future_bag=self._new_bag()   # ← 2 本目
        self.current=None
        self.hold_piece=None; self.can_hold=True
        self.lock_timer=None; self.game_over=False; self.tick=0
        self.move_state={k:{'held':False,'ts':0}
                         for k in (pg.K_LEFT,pg.K_RIGHT,pg.K_DOWN)}
        self.ai_mode=ai_mode; self.target_rot=0; self.target_x=4
        self._prev_was_tetris = False
        self.spawn()

    # 7 種シャッフルバッグ
    def _new_bag(self):
        b=list('IOTSZJL'); random.shuffle(b); return b

    # これから落ちてくる n 個を返す
    def next_queue(self,n:int)->list[str]:
        return (self.bag[::-1] + self.future_bag[::-1])[:n]

    # 次ミノ取り出し
    def _next(self):
        if not self.bag:
            self.bag, self.future_bag = self.future_bag, self._new_bag()
        return self.bag.pop()

    def spawn(self):
        # ① 新しい現在ミノを取得
        self.current = Piece(self._next())

        # ② 出現位置がすでに詰まっていたらゲームオーバー
        if not self.board.valid(self.current, 0):
            self.game_over = True
            return

        # ③ ホールド可・ロックタイマー初期化
        self.can_hold   = True
        self.lock_timer = None

        # ④ AI 用ターゲット計算（Next-5 先読み）
        if self.ai_mode:
            next4 = self.next_queue(4)                 # これから落ちる 4 個
            move  = AI.best_move(self.board,
                                self.current.kind,
                                next4)                # ← 追加引数
            # best_move が None の場合はその場でハードドロップさせないよう保険
            self.target_rot, self.target_x = move if move else (0, self.current.x)


    # ── 移動/ドロップ/保持 ─────────────────────
    def side(self,dx): self.current.x+=dx if self.board.valid(self.current,self.current.rot,dx,0) else 0
    def soft(self):    self.current.y-=1 if self.board.valid(self.current,self.current.rot,0,-1) else 0
    def hard(self):
        while self.board.valid(self.current,self.current.rot,0,-1): self.current.y-=1
        self._lock()
    def hold(self):
        if not self.can_hold: return
        self.can_hold=False
        if self.hold_piece is None:
            self.hold_piece=Piece(self.current.kind); self.spawn()
        else:
            self.current.kind,self.hold_piece.kind=self.hold_piece.kind,self.current.kind
            self.current.x,self.current.y,self.current.rot=4,21,0
            if not self.board.valid(self.current,0): self.game_over=True
    def _lock(self):
        self.board.lock(self.current)
        cleared = self.board.clear()
        # クリア枚数が0以上のときだけ表示
        if cleared > 0:
            # 今回がテトリス（4ライン）かどうか
            is_tetris = (cleared == 4)
            # かつ前回もテトリスだったら B2B
            b2b = (is_tetris and self._prev_was_tetris)
            tag = " B2B" if b2b else "  nonB2B"
            print(f"[_lock] cleared {cleared} lines{tag}")
            # 次回の判定用にフラグ更新
            self._prev_was_tetris = is_tetris
        
        # 盤面外にはみ出すセルがあったらアウト
        out_of_bounds = any(y >= MATRIX_H for _, y in self.current.cells())

        if out_of_bounds:
            self.game_over = True
        else:
            self.spawn()

        

    # ── 重力 ──────────────────────────────────
    def gravity(self):
        if self.board.valid(self.current,self.current.rot,0,-1):
            self.current.y-=1; self.lock_timer=None
        else:
            if self.lock_timer is None: self.lock_timer=pg.time.get_ticks()
            elif pg.time.get_ticks()-self.lock_timer>=LOCK_DELAY_MS: self._lock()

    # ── 入力処理 ──────────────────────────────
    def handle(self,e:pg.event.Event):
        if e.type==pg.KEYDOWN and e.key==pg.K_ESCAPE: self.game_over=True
        if e.key in self.move_state:
            if e.type==pg.KEYDOWN:
                if e.key==pg.K_DOWN: self.soft()
                else: self.side(-1 if e.key==pg.K_LEFT else 1)
                st=self.move_state[e.key]; st['held']=True; st['ts']=pg.time.get_ticks()
            elif e.type==pg.KEYUP: self.move_state[e.key]['held']=False
        elif e.type==pg.KEYDOWN:
            if e.key in (pg.K_z,pg.K_q,pg.K_LCTRL,pg.K_RCTRL): self.current.rotate(-1,self.board)
            elif e.key in (pg.K_x,pg.K_w,pg.K_UP):            self.current.rotate(1,self.board)
            elif e.key==pg.K_SPACE: self.hard()
            elif e.key==pg.K_c:     self.hold()

    def auto_repeat(self):
        now=pg.time.get_ticks()
        for k,s in self.move_state.items():
            if s['held'] and now-s['ts']>=MOVE_DELAY_MS:
                if k==pg.K_LEFT: self.side(-1)
                elif k==pg.K_RIGHT: self.side(1)
                elif k==pg.K_DOWN: self.soft()
                s['ts']+=MOVE_REPEAT_MS

    def ai_step(self):
        if not self.ai_mode: return
        if self.current.rot!=self.target_rot:
            self.current.rotate(1,self.board); return
        if self.current.x<self.target_x: self.side(1); return
        if self.current.x>self.target_x: self.side(-1); return
        self.hard()

    def update(self):
        self.tick+=1
        if self.tick%GRAVITY[0]==0: self.gravity()

# ───────────────── Renderer ─────────────────────
class Renderer:
    def __init__(self,g:Game):
        self.g=g; w=BLOCK*(MATRIX_W+10); h=BLOCK*(MATRIX_H+2)
        self.sc=pg.display.set_mode((w,h)); pg.display.set_caption('Tetris')
        self.font=pg.font.SysFont('consolas',18)
    def _cell(self,x,y,c,a=255):
        if 0<=y<MATRIX_H:
            r=pg.Rect(x*BLOCK,(MATRIX_H-1-y)*BLOCK,BLOCK,BLOCK)
            s=pg.Surface((BLOCK-1,BLOCK-1)); s.fill(c); s.set_alpha(a)
            self.sc.blit(s,(r.x+1,r.y+1))
    def _mini(self,k,ox,oy):
        for dx, dy in PIECE_SHAPES[k][0]:
            # S/Z に加えて L/J も上下反転
            if k in ('S', 'Z', 'L', 'J'):
                dy = -dy
            pg.draw.rect(self.sc,COLORS[k],
                         (ox+dx*BLOCK//2, oy+dy*BLOCK//2, BLOCK//2, BLOCK//2))

    def draw(self):
        g=self.g; self.sc.fill((0,0,0))
        pg.draw.rect(self.sc,(40,40,40),(0,0,BLOCK*MATRIX_W,BLOCK*MATRIX_H))

        # 固定ブロック
        for y,row in enumerate(g.board.grid):
            for x,c in enumerate(row):
                if c: self._cell(x,y,COLORS[c])

        # ゴースト
        gy=g.current.y
        while g.board.valid(g.current,g.current.rot,0,gy-g.current.y-1): gy-=1
        for x,y in g.current.cells():
            self._cell(x,gy-(g.current.y-y),COLORS['G'],80)

        # 現在ミノ
        for x,y in g.current.cells(): self._cell(x,y,COLORS[g.current.kind])

        # Hold
        bx=BLOCK*(MATRIX_W+1)
        pg.draw.rect(self.sc,(100,100,100),(bx,BLOCK*2,BLOCK*4,BLOCK*4),2)
        self.sc.blit(self.font.render('HOLD',True,(200,200,200)),(bx,BLOCK))
        if g.hold_piece: self._mini(g.hold_piece.kind,bx+BLOCK,BLOCK*3)

        # Next-5
        self.sc.blit(self.font.render('NEXT',True,(200,200,200)),(bx,BLOCK*6))
        for i,k in enumerate(g.next_queue(5)):
            ox,oy=bx,BLOCK*(7+i*3)
            pg.draw.rect(self.sc,(100,100,100),(ox,oy,BLOCK*4,BLOCK*4),2)
            self._mini(k,ox+BLOCK,oy+BLOCK)

        pg.display.flip()

# ─────────────────── main ───────────────────────
def main():
    pg.init(); clock=pg.time.Clock()
    game=Game(ai_mode=True)   # True で AI プレイ True False
    rndr=Renderer(game)
    while not game.game_over:
        for e in pg.event.get():
            if e.type==pg.QUIT: return
            if e.type in (pg.KEYDOWN,pg.KEYUP): game.handle(e)
        game.auto_repeat(); game.ai_step(); game.update(); rndr.draw()
        clock.tick(FPS)
    pg.quit()

if __name__=='__main__':
    main()