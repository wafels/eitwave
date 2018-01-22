#
# test using SymPy to do the algebra deriving the
# velocity-acceleration algebra
#
from sympy import *

# Time
t = symbols('t')

# Measured arc distance
y = symbols('y')

# Initial distance, velocity and acceleration
S = symbols('S')
V = symbols('V')
A = symbols('A')

# Distance traveled
D = S + V*t + 0.5*A*t*t

# Minimization
diff = (y - D)**2

# Expand the expression, and collect the expression
# in powers of time
dc = expand(diff)

# Small changes to the initial distance, velocity and acceleration
s = symbols('s')
v = symbols('v')
a = symbols('a')

# Distance travelled
D1 = S + s + (V+v)*t + 0.5*(A+a)*t*t

# Minimization
diff1 = (y - D1)**2

# Expand the expression, and collect the expression
# in powers of time
dc1 = expand(diff1)

# Assume that each minimization gives approximately the same
# result.  Therefore the difference is zero, so we subtract
# one from the other, simplify it, and collect in powers of
expr = expand(dc - dc1)

# Remove the initial displacements
e2 = (expr.subs(S, 0)).subs(s, 0)

# Collect as a function of velocity
e3 = collect(expr, v)

# Render as Latex
latex(e3)
