from dtl_algebra import (
    generate_dtl_states,
    apply_identity,
    apply_half_vacuum_1,
    apply_half_vacuum_2,
    apply_half_vacuum_3,
    apply_half_vacuum_4,
    apply_tl
)
import sys

def demo_actions(L=4, j=0):
    states = generate_dtl_states(L, j)
    print(f"Module V^{{{L},{j}}} has {len(states)} states:")
    for idx, s in enumerate(states):
        print(f"|{idx}> = {s}")
    print("\n" + "="*50 + "\n")

    n = "n"  # We'll just use a string literal 'n' to represent the weight for clarity

    generators = [
        ("Identity", apply_identity),
        ("Half-Vacuum Type 1", apply_half_vacuum_1),
        ("Half-Vacuum Type 2", apply_half_vacuum_2),
        ("Half-Vacuum Type 3", apply_half_vacuum_3),
        ("Half-Vacuum Type 4", apply_half_vacuum_4),
        ("Temperley-Lieb", apply_tl)
    ]

    for gen_name, gen_func in generators:
        print(f"--- Action of {gen_name} ---")
        for i in range(L - 1):
            print(f"  acting on sites ({i}, {i+1}):")
            for idx, state in enumerate(states):
                if gen_name == "Temperley-Lieb":
                    res = gen_func(state, i, n)
                else:
                    res = gen_func(state, i)

                if not res:
                    print(f"    {gen_name}_{i} |{idx}> = 0")
                else:
                    output_str = " + ".join([f"{weight} * {s}" for s, weight in res])
                    print(f"    {gen_name}_{i} |{idx}> = {output_str}")
        print("\n")

if __name__ == "__main__":
    L = 4
    j = 0
    if len(sys.argv) > 2:
        L = int(sys.argv[1])
        j = int(sys.argv[2])
    demo_actions(L, j)
