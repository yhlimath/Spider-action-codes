(* Diagonalization Analysis for S=5, x=1, y=2 *)

OriginalBasis = {
{1, 1, 1, 0, 0},
{1, 1, 0, 1, 0},
{1, 1, 0, 0, 1},
{1, 0, 1, 1, 0},
{1, 0, 1, 0, 1}
};

ActiveIndices = {2};

OriginalMatrix = {
{0, 0, 0, 0, 0},
{n, 1, n, n, n^2},
{0, 0, 0, 0, 0},
{0, 0, 0, 0, 0},
{0, 0, 0, 0, 0}
};

RestrictedSubmatrix = {
{1}
};

DiagonalizedMatrix = {
{1}
};

EigenvectorBasisSubmatrix = {
{1}
};

FullBasisTransformationMatrix = {
{1, 0, 0, 0, 0},
{0, 1, 0, 0, 0},
{0, 0, 1, 0, 0},
{0, 0, 0, 1, 0},
{0, 0, 0, 0, 1}
};
