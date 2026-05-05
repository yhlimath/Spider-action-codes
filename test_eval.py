from denseKuperberg.transfer_matrix import evaluate_coeff
from sl3hecke.sl3_hecke import Polynomial

c1 = Polynomial({1: 1.0})
print("Poly eval:", evaluate_coeff(c1, 1.0))
print("Float eval:", evaluate_coeff(0.5, 1.0))
