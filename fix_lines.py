import re

for filename in ['neural_lam/metrics.py', 'neural_lam/models/ar_model.py']:
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    out = []
    for line in lines:
        if len(line.rstrip()) > 79 and "    -" in line and "One of " not in line:
            # e.g. "    - pred: Tensor of shape [..., N_grid, d_features], predictions"
            idx = line.find(':')
            if idx != -1 and idx < 75:
                # split at colon
                out.append(line[:idx+1] + "\n")
                # Add spaces
                out.append(" " * (line.find('-') + 2) + line[idx+1:].lstrip())
            else:
                out.append(line)
        elif len(line.rstrip()) > 79 and "One of (...,)" in line:
            # this is metrics.py line 35
            out.append(line.replace("), depending", "),\n      depending"))
        elif len(line.rstrip()) > 79 and "sum_vars:" in line:
            out.append(line.replace(" (sum", "\n      (sum"))
        elif len(line.rstrip()) > 79 and "average_grid:" in line:
            out.append(line.replace(" (mean", "\n      (mean"))
        elif len(line.rstrip()) > 79 and "Tensor(" in line:
            out.append(line.replace(", shape", ",\n        shape"))
        elif len(line.rstrip()) > 79 and "pred" in line and "Tensor" in line:
             out.append(line.replace("), predicted", "),\n        predicted"))
        elif len(line.rstrip()) > 79 and "prediction" in line:
             out.append(line.replace(", prediction", ",\n        prediction"))
        elif len(line.rstrip()) > 79 and "target" in line:
             out.append(line.replace(", target", ",\n        target"))
        elif len(line.rstrip()) > 79:
             # Just roughly break them if they contain a comma
             r_idx = line.rfind(',', 0, 75)
             if r_idx != -1:
                 out.append(line[:r_idx+1] + "\n")
                 out.append(" " * (line.find('-') + 2 if '-' in line else 8) + line[r_idx+1:].lstrip())
             else:
                 out.append(line)
        else:
            out.append(line)
            
    with open(filename, 'w') as f:
        f.writelines(out)

