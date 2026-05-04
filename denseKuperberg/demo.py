import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseKuperberg.states import generate_paths
from denseKuperberg.algebra import action_E_i, action_H_i

def print_path(path):
    return "[" + ", ".join([f"({s:2}, {j:2})" for s, j in path]) + "]"

def main():
    L = 3
    x, y = 0, 0
    print(f"Generating paths for L={L}, x={x}, y={y}...")
    paths = generate_paths(L, x, y)
    print(f"Found {len(paths)} paths.")
    for i, p in enumerate(paths):
        print(f"Path {i}: {print_path(p)}")

    print("\n--- Testing E_i Actions ---")
    for i, p in enumerate(paths):
        print(f"\nOriginal Path {i}: {print_path(p)}")
        for step_idx in range(len(p) - 1):
            res = action_E_i(1, p, step_idx)
            if res:
                print(f"  E_{step_idx} applied:")
                for coeff, new_p in res:
                    print(f"    Coeff: {coeff}, Path: {print_path(new_p)}")
            else:
                print(f"  E_{step_idx} applied: 0")

    print("\n--- Testing H_i Actions ---")
    for i, p in enumerate(paths):
        print(f"\nOriginal Path {i}: {print_path(p)}")
        for step_idx in range(len(p) - 1):
            try:
                res = action_H_i(1, p, step_idx)
                if res:
                    print(f"  H_{step_idx} applied:")
                    for coeff, new_p in res:
                        print(f"    Coeff: {coeff}, Path: {print_path(new_p)}")
                else:
                    print(f"  H_{step_idx} applied: 0")
            except Exception as e:
                print(f"  H_{step_idx} applied: Error - {e}")

if __name__ == "__main__":
    main()
