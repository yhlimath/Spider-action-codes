from denseKuperberg.algebra import action_TLD_i
path = [(1, 1), (-1, 0), (-1, -1)]
res = action_TLD_i(1, path, 0)
for c, p in res:
    print(p)
