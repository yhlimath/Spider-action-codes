(* Diagonalization Analysis for S=4, x=1, y=0 *)

OriginalBasis = {
{1, 1, 0, -1},
{1, 0, 1, -1},
{1, 0, -1, 1}
};

ActiveIndices = {1, 3};

OriginalMatrix = {
{n, n^2, n},
{0, 0, 0},
{n, n^2, n}
};

RestrictedSubmatrix = {
{n, n},
{n, n}
};

DiagonalizedMatrix = {
{0, 0},
{0, 2*n}
};

EigenvectorBasisSubmatrix = {
{-1, 1},
{1, 1}
};

FullBasisTransformationMatrix = {
{-1, 0, 1},
{0, 1, 0},
{1, 0, 1}
};
