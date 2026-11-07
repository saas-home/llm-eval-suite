class _N:
    __slots__ = ("k", "l", "r", "h")
    def __init__(self, k):
        self.k, self.l, self.r, self.h = k, None, None, 1

def _h(n): return n.h if n else 0
def _fix(n): n.h = 1 + max(_h(n.l), _h(n.r))
def _rr(y):
    x = y.l; y.l = x.r; x.r = y; _fix(y); _fix(x); return x
def _rl(x):
    y = x.r; x.r = y.l; y.l = x; _fix(x); _fix(y); return y
def _bal(n):
    _fix(n); b = _h(n.l) - _h(n.r)
    if b > 1:
        if _h(n.l.l) < _h(n.l.r): n.l = _rl(n.l)
        return _rr(n)
    if b < -1:
        if _h(n.r.r) < _h(n.r.l): n.r = _rr(n.r)
        return _rl(n)
    return n

class AVLTree:
    def __init__(self): self.root = None
    def insert(self, k): self.root = self._ins(self.root, k)
    def _ins(self, n, k):
        if not n: return _N(k)
        if k < n.k: n.l = self._ins(n.l, k)
        elif k > n.k: n.r = self._ins(n.r, k)
        else: return n
        return _bal(n)
    def delete(self, k): self.root = self._del(self.root, k)
    def _del(self, n, k):
        if not n: return None
        if k < n.k: n.l = self._del(n.l, k)
        elif k > n.k: n.r = self._del(n.r, k)
        else:
            if not n.l: return n.r
            if not n.r: return n.l
            m = n.r
            while m.l: m = m.l
            n.k = m.k; n.r = self._del(n.r, m.k)
        return _bal(n)
    def search(self, k):
        n = self.root
        while n:
            if k == n.k: return True
            n = n.l if k < n.k else n.r
        return False
    def height(self): return _h(self.root)
    def __iter__(self):
        st, n = [], self.root
        while st or n:
            while n: st.append(n); n = n.l
            n = st.pop(); yield n.k; n = n.r
