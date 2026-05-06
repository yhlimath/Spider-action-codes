import sys
import re

with open('denseKuperberg/algebra.py', 'r') as f:
    content = f.read()

# Remove T1 and T2
content = re.sub(r'def action_T1_x_i.*?(?=def action_T2_x_i)', '', content, flags=re.DOTALL)
content = re.sub(r'def action_T2_x_i.*?(?=\n\n|\Z)', '', content, flags=re.DOTALL)

# Add T_xyz_i
new_action = """
def action_T_xyz_i(coeff, path, i, x, y, z, n_value):
    \"\"\"
    The universally parameterized action:
    T_i(x,y,z) = x*I + y*TL_i + z*(E_i + H_i)
    \"\"\"
    from sl3hecke.sl3_hecke import Polynomial

    result = []

    # x * I
    if x != 0:
        cx = coeff * x if not isinstance(coeff, Polynomial) else coeff * x
        result.extend(action_ID_i(cx, path, i))

    # y * TL_i
    if y != 0:
        cy = coeff * y if not isinstance(coeff, Polynomial) else coeff * y
        result.extend(action_TL_i(cy, path, i))

    # z * (E_i + H_i)
    if z != 0:
        cz = coeff * z if not isinstance(coeff, Polynomial) else coeff * z
        result.extend(action_E_i(cz, path, i))
        result.extend(action_H_i(cz, path, i))

    return result
"""

with open('denseKuperberg/algebra.py', 'w') as f:
    f.write(content.strip() + new_action)
