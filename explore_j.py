from dilute_temperley_lieb.dtl_transfer_matrix import construct_dtl_transfer_matrix

L = 10
for j in range(L + 1):
    try:
        T, states = construct_dtl_transfer_matrix(L, j)
        print(f"dTL L={L}, j={j} -> dim = {len(states)}")
    except Exception as e:
        print(f"dTL L={L}, j={j} -> Error")
