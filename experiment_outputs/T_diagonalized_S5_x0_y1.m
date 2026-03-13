(* Diagonalization Analysis for S=5, x=0, y=1 *)

OriginalBasis = {
{1, 1, 0, 0, -1},
{1, 1, 0, -1, 0},
{1, 0, 1, 0, -1},
{1, 0, 1, -1, 0},
{1, 0, -1, 1, 0}
};

ActiveIndices = {1, 5};

OriginalMatrix = {
{n^2 + 1, n, n^3, n^2, n},
{0, 0, 0, 0, 0},
{0, 0, 0, 0, 0},
{0, 0, 0, 0, 0},
{n, n^2, 0, n^3, n^2}
};

RestrictedSubmatrix = {
{n^2 + 1, n},
{n, n^2}
};

DiagonalizedMatrix = {
{n^2 - 1/2*(4*n^2 + 1)^(1/2) + 1/2, 0},
{0, n^2 + (1/2)*(4*n^2 + 1)^(1/2) + 1/2}
};

EigenvectorBasisSubmatrix = {
{(1/2)*(1 - (4*n^2 + 1)^(1/2))/n, (1/2)*((4*n^2 + 1)^(1/2) + 1)/n},
{1, 1}
};

FullBasisTransformationMatrix = {
{(1/2)*(1 - (4*n^2 + 1)^(1/2))/n, 0, 0, 0, (1/2)*((4*n^2 + 1)^(1/2) + 1)/n},
{0, 1, 0, 0, 0},
{0, 0, 1, 0, 0},
{0, 0, 0, 1, 0},
{1, 0, 0, 0, 1}
};
