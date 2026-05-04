import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from denseKuperberg.states import generate_paths
from denseKuperberg.algebra import action_E_i, action_H_i

def print_path(path):
    return "[" + ", ".join([f"({s:2}, {j:2})" for s, j in path]) + "]"

def main():
    L = 6
    x, y = 0, 0
    print(f"Generating paths for L={L}, x={x}, y={y}...")
    paths = generate_paths(L, x, y)
    print(f"Found {len(paths)} paths.")

    print("\n--- Testing actions that produce non-zero results ---")
    e_success = 0
    h_success = 0
    for i, p in enumerate(paths):
        for step_idx in range(len(p) - 1):
            res_e = action_E_i(1, p, step_idx)
            if res_e:
                e_success += 1
                if e_success <= 2:
                    print(f"Path: {print_path(p)}")
                    print(f"  E_{step_idx} applied:")
                    for coeff, new_p in res_e:
                        print(f"    Coeff: {coeff}, Path: {print_path(new_p)}")

            try:
                res_h = action_H_i(1, p, step_idx)
                if res_h:
                    h_success += 1
                    if h_success <= 2:
                        print(f"Path: {print_path(p)}")
                        print(f"  H_{step_idx} applied:")
                        for coeff, new_p in res_h:
                            print(f"    Coeff: {coeff}, Path: {print_path(new_p)}")
            except Exception as e:
                pass

    print(f"\nTotal non-zero E applications: {e_success}")
    print(f"Total non-zero H applications: {h_success}")

if __name__ == "__main__":
    main()
