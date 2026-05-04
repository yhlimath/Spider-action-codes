def step_weight(s, j):
    """
    Returns the weight (x, y) of a step u_{s, j}.
    s in {1, -1} (for +, -)
    j in {1, 0, -1}

    Rules:
    u_{+,1} = lambda_1 = (1, 0)
    u_{+,0} = lambda_2 - lambda_1 = (-1, 1)
    u_{+,-1} = -lambda_2 = (0, -1)
    u_{-s,-j} = -u_{s,j}
    """
    if s == 1:
        if j == 1: return (1, 0)
        elif j == 0: return (-1, 1)
        elif j == -1: return (0, -1)
    elif s == -1:
        # u_{-, j} = -u_{+, -j}
        weight_pos = step_weight(1, -j)
        return (-weight_pos[0], -weight_pos[1])
    raise ValueError(f"Invalid step: s={s}, j={j}")

def generate_paths(L, target_x, target_y):
    """
    Generates all valid paths P = (S, J) of length L with final weight (target_x, target_y).
    Returns a list of paths, where each path is a list of tuples (s, j).
    """
    results = []

    def backtrack(path, current_x, current_y):
        if len(path) == L:
            if current_x == target_x and current_y == target_y:
                results.append(path[:])
            return

        for s in [1, -1]:
            for j in [1, 0, -1]:
                dx, dy = step_weight(s, j)
                new_x = current_x + dx
                new_y = current_y + dy

                # Partial sums must have x_l >= 0 and y_l >= 0
                if new_x >= 0 and new_y >= 0:
                    path.append((s, j))
                    backtrack(path, new_x, new_y)
                    path.pop()

    backtrack([], 0, 0)
    return results
