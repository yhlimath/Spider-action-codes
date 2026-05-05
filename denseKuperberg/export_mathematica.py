import json
import os
import math

def format_complex_mma(real, imag):
    if abs(imag) < 1e-12:
        return f"{real:.10f}"
    if imag >= 0:
        return f"{real:.10f} + I*{imag:.10f}"
    return f"{real:.10f} - I*{abs(imag):.10f}"

def export_logs_to_mma():
    in_file = "experiment_outputs/denseKuperberg/eigenvalue_logs_top_k.json"
    if not os.path.exists(in_file):
        print(f"Log file {in_file} not found.")
        return

    with open(in_file, 'r') as f:
        logs = json.load(f)

    out_dir = "experiment_outputs/denseKuperberg"
    os.makedirs(out_dir, exist_ok=True)

    # Write a .m data file
    m_file = os.path.join(out_dir, "eigenvalues_top_k.m")
    with open(m_file, 'w') as f:
        f.write("(* Mathematica format eigenvalue logs *)\n")
        f.write("(* Usage: ev[L][\"type\"][\"order\"][\"n\"][j] *)\n")
        f.write("ClearAll[ev];\n\n")

        for t in logs:
            for order in logs[t]:
                for n_str in logs[t][order]:
                    for L_str, lam_list in logs[t][order][n_str].items():
                        L = int(L_str)
                        if lam_list is not None:
                            # Write each eigenvalue index
                            for j, lam in enumerate(lam_list):
                                val_str = format_complex_mma(lam['real'], lam['imag'])
                                f.write(f"ev[{L}][\"{t}\"][\"{order}\"][\"{n_str}\"][{j}] = {val_str};\n")

    print(f"Exported data to {m_file}")

    # Write a Mathematica notebook interface string to a plain text file.
    # The user can paste this into Mathematica.
    nb_interface_file = os.path.join(out_dir, "conformal_extrapolation_interface.txt")
    with open(nb_interface_file, 'w') as f:
        f.write("""(* Paste this code into a Mathematica notebook after running Get["eigenvalues_top_k.m"] *)

(* Configurable Fermi Velocity *)
vF = 1.0;

(* Function to compute conformal dimension from Transfer Matrix formulation *)
(* h_j = (L / (\\[Pi] vF)) * Log[ Abs[ev[L][t][order][n][0]] / Abs[ev[L][t][order][n][j]] ] *)
CalcH[L_, t_, order_, n_, j_] := Module[{lam0, lamj},
  lam0 = ev[L][t][order][n][0];
  lamj = ev[L][t][order][n][j];
  If[NumericQ[lam0] && NumericQ[lamj],
     (L / (Pi * vF)) * Log[ Abs[lam0] / Abs[lamj] ],
     Missing["NotAvailable"]
  ]
];

(* Interactive Panel for Tracking and Extrapolating Eigenvalues *)
(* This panel allows independent selection of eigenvalue indices j for each size L to track wall-crossings. *)
Manipulate[
  Module[{Llist, jvals, hvals, fitData, L, fitLine, x},
    Llist = {3, 4, 5, 6};
    jvals = {j3, j4, j5, j6};

    hvals = Table[{Llist[[i]], CalcH[Llist[[i]], type, order, n, jvals[[i]]]}, {i, 1, Length[Llist]}];
    hvals = Select[hvals, NumericQ[#[[2]]]&];

    If[Length[hvals] >= 2,
      (* Fit against 1/L^2 as standard for boundary conformal bounds *)
      fitData = {1/(#[[1]]^2), #[[2]]}& /@ hvals;
      fitLine = Fit[fitData, {1, x}, x];

      Column[{
        Text[Style["Extrapolated Conformal Dimension (L -> \\[Infinity]): " <> ToString[fitLine /. x -> 0], Bold, 14]],
        ListPlot[fitData,
          PlotRange -> All,
          AxesLabel -> {"1/L^2", "h_j(L)"},
          Epilog -> {Red, Line[Table[{x, fitLine}, {x, 0, Max[fitData[[All, 1]]], 0.01}]]},
          ImageSize -> Large
        ]
      }],
      Text["Not enough data points for the selected indices."]
    ]
  ],
  {{type, "E+H", "Matrix Type"}, {"E+H+H2", "E+H", "H2"}},
  {{order, "sequential", "Order"}, {"sequential", "staggered"}},
  {{n, "1.0", "Weight (n)"}, {"0.5", "0.75", "0.8", "1.0", "1.2", "1.25", "1.5", "1.75", "1.8", "2.0"}},
  {{j3, 1, "Index j for L=3"}, 0, 49, 1},
  {{j4, 1, "Index j for L=4"}, 0, 49, 1},
  {{j5, 1, "Index j for L=5"}, 0, 49, 1},
  {{j6, 1, "Index j for L=6"}, 0, 49, 1}
]
""")

    print(f"Exported Mathematica GUI tool text to {nb_interface_file}")

if __name__ == "__main__":
    export_logs_to_mma()
