import math
import mpmath
# ---------------------------------------------------------------------
#  Low-level building blocks
# ---------------------------------------------------------------------
_SPLITTER = 134217729.0          # 2^27+1, exact in double

def _split(a: float):
    """Dekker split:  a  ->  (ahi, alo) such that ahi+alo == a  and
       each part has <= 27 significant bits so their product fits."""
    c      = _SPLITTER * a
    abig   = c - a
    ahi    = c - abig
    alo    = a - ahi
    return ahi, alo

def _two_sum(a: float, b: float):
    """Accurate sum: returns (s, err) with  s+err == a+b  exactly."""
    s   = a + b
    bb  = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err

def _quick_two_sum(a: float, b: float):
    """Assumes |a| ≥ |b|.  Slightly faster version of _two_sum."""
    s   = a + b
    err = b - (s - a)
    return s, err

def _two_prod(a: float, b: float):
    """Accurate product: returns (p, err) with  p+err == a*b  exactly."""
    p = a * b
    # use FMA if available (gives exact error in one step)
    if hasattr(math, "fma"):
        err = math.fma(a, b, -p)
        return p, err
    # fallback: Dekker splitting
    ahi, alo = _split(a)
    bhi, blo = _split(b)
    err = ((ahi*bhi - p) + ahi*blo + alo*bhi) + alo*blo
    return p, err

# ---------------------------------------------------------------------
#  Double-double number and basic ops
# ---------------------------------------------------------------------
class dd(tuple):
    """Store as (hi, lo)     invariant: |lo| ≤ 0.5 ulp(hi)."""
    __slots__ = ()

    # construction helpers ------------------------------------------------
    def __new__(cls, hi: float, lo: float = 0.0):
        return super().__new__(cls, (float(hi), float(lo)))

    @staticmethod
    def from_double(a: float):              # promote one double
        return dd(float(a), 0.0)
    @staticmethod
    def from_str(s: str):
        """Convert high-precision string to double-double by rounding."""
        mpmath.prec = 160
        x = mpmath.mpf(s)
        hi = float(x)                         # first round
        lo = float(x - hi)                    # remaining part
        return dd(hi, lo)

    # high-precision elementary operations --------------------------------
    def __add__(self, other):
        if not isinstance(other, dd): other = dd.from_double(other)
        s, e  = _two_sum(self[0], other[0])        # add hi parts
        t, f  = _two_sum(self[1], other[1])        # add lo parts
        e    += t                                  # collect low error
        s2, e2 = _quick_two_sum(s, e)              # renormalise
        lo     = e2 + f
        hi, lo = _quick_two_sum(s2, lo)
        return dd(hi, lo)

    def __sub__(self, other):
        if not isinstance(other, dd): other = dd.from_double(other)
        return self.__add__(dd(-other[0], -other[1]))

    def __mul__(self, other):
        if not isinstance(other, dd): other = dd.from_double(other)
        p, e   = _two_prod(self[0], other[0])      # high product
        e      = e + (self[0]*other[1] + self[1]*other[0])
        hi, lo = _quick_two_sum(p, e)              # renormalise
        return dd(hi, lo)

    def __truediv__(self, other):
        if not isinstance(other, dd): other = dd.from_double(other)

        # long division:  q  ~=  hi/other.hi
        q1      = self[0] / other[0]
        q1_dd   = dd.from_double(q1)

        # compute remainder  r = self - q1*other   in double-double
        r       = self - other * q1_dd

        # refine: q2 = r.hi / other.hi
        q2      = r[0] / other[0]
        q2_dd   = dd.from_double(q2)

        return q1_dd + q2_dd                       # q = q1+q2

    # convenience ----------------------------------------------------------
    def __repr__(self):
        return f"dd(hi={self[0]:.17g}, lo={self[1]:.17g})"

    def to_double(self):
        """Return closest double (just hi)."""
        return self[0] + self[1]

# -------------------------------------------------------------------------
#  The requested expression
# -------------------------------------------------------------------------
def expr(a: str, b: str) -> dd:
    """Compute  (a*b + 1) / (b - a)  in double-double precision."""
    a_dd = dd.from_str(a)
    b_dd = dd.from_str(b)

    numerator   = a_dd * b_dd + dd.from_str("1.0")
    denominator = b_dd - a_dd
    if denominator[0] == 0.0 and denominator[1] == 0.0:
        raise ZeroDivisionError("b == a  →  denominator is zero")

    return numerator #/ denominator

def high_precision_expr(a, b, prec=106):
    """
    Computes (a * b + 1) / (b - a) using mpmath with arbitrary precision.

    Parameters:
    - a, b: float or str — input values.
    - prec: int — number of bits of precision (default: 106 ≈ double-double).

    Returns:
    - mpmath.mpf: high-precision result.
    """
    mpmath.mp.prec = prec  # set precision in *bits*

    a_mp = mpmath.mpf(a)
    b_mp = mpmath.mpf(b)

    numerator   = a_mp * b_mp + mpmath.mpf("1")
    denominator = b_mp - a_mp

    if denominator == 0:
        raise ZeroDivisionError("b == a → denominator is zero")

    return numerator #/ denominator

# -------------------------------------------------------------------------
#  Tiny demo
# -------------------------------------------------------------------------
if __name__ == "__main__":
    a = "1.234567890123456"
    b = "1.234567890123457"       # nearly identical → catastrophic cancellation

    # naive  = (a*b + 1.0)#/(b - a)
    high   = expr(a,b).to_double()
    result = high_precision_expr(a, b)

    # print(f"naive  (double)      : {naive:.20f}")
    print(f"double-double result : {high :.20f}")
    print(f"expression: {result}")