
DECIMAL_PRECISION = 10

# ---------------------------- MATRIX OPERATIONS -------------------------------


def transposed(matrix):

    return [[matrix[line][column] for line in range(len(matrix))] for column in range(len(matrix[0]))]


def minor(matrix, omitted_line, omitted_column):

    return [matrix[line][:omitted_column] + matrix[line][omitted_column + 1:] for line in range(omitted_line)] + [matrix[line][:omitted_column] + matrix[line][omitted_column + 1:] for line in range(omitted_line + 1, len(matrix))]


def determinant(matrix):

    if len(matrix) == 1:
        return matrix[0][0]

    # aplying Laplace formula on the first line

    det = 0
    sgn = 1

    for col in range(len(matrix[0])):

        det += sgn * matrix[0][col] * determinant(minor(matrix, 0, col))
        sgn *= -1

    return det


def inverse(matrix):

    det = determinant(matrix)

    if det == 0:
        raise Exception("NOT INVERTIBLE")

    # calculating the cofactor matrix

    cofactor_matrix = [[0 for j in range(len(matrix[0]))] for i in range(len(matrix))]

    for l in range(len(matrix)):
        for c in range(len(matrix[0])):

            cofactor_matrix[l][c] = ((-1) ** (l + c)) * determinant(minor(matrix, l, c))

    for l in range(len(cofactor_matrix)):
        for c in range(len(cofactor_matrix[0])):

            cofactor_matrix[l][c] /= det
            cofactor_matrix[l][c] = round(cofactor_matrix[l][c], DECIMAL_PRECISION)

    return transposed(cofactor_matrix)


def multiply(a, b):

    # matrix a included in M(m, n)R
    # matrix b included in M(n, p)R
    # matrix product included in M(m, p)R

    m = len(a)
    n = len(b)
    p = len(b[0])

    product_matrix = [[0 for j in range(p)] for i in range(m)]

    for line in range(m):
        for column in range(p):
            for k in range(n):

                product_matrix[line][column] += a[line][k] * b[k][column]

    return product_matrix


# -------------------------------------------------------------------------------

# general description

print('\nmultiple linear regression software\n')

print('describing the input file:\n')

print('first line: number of independent variables - k')
print('second line: number of observations - n')
print('next n lines: tuples of data corresponding to each observation: x0 x1 x2 ... xk y')

# processing input from input file

input_file = open('input_data.txt')

k = input_file.readline()
k = int(k[:len(k) - 1])

n = input_file.readline()
n = int(n[:len(n) - 1])

test = []

for i in range(n):

    s_test = input_file.readline()

    test.append(tuple(float(c) for c in s_test.split()))

print(f'\ninput processed: k = {k}, n = {n}, observations are: {test}')

# calculating coefficients b0, b1, b2, ... bk

# we want bi coefficients so that SUM(i = 0, n) (Yi - (b0 + b1 * xi1 + b2 * xi2 + ... bk * xik)) ^ 2 is minimal
# all the xij s and yi s are known, except for parameters b0, b1, ... bk
# because of that, we treat this expression as a function S: R^k -> R
# the problem is reduced to finding the minimum point of this function
# minimum point satisfies the following: dS / dbi = 0 for all bi with i = 1, k
# after calculating all the partial derivates, we obtain a linear system of equations, with unknowns b0, ... bk
# the linear system can be expressed with matrices and vector multiplication:
# transposed(X) * X * B = trasposed(X) * Y => B = inverse((transposed(X) * X)) * transposed(X) * Y

# NOTE: program is NOT CONFIGURED to work when (transposed(X) * X) is NOT invertible

print('\ncalculating coefficients...')

# configuring matrix X

X = [[0 for j in range(k + 1)] for i in range(n)]

for i_test in range(n):

    X[i_test][0] = 1

    for coord in range(len(test[i_test]) - 1):

        X[i_test][coord + 1] = test[i_test][coord]

# configuring matrix Y

Y = [[test[i_test][k]] for i_test in range(n)]

print(f'auxiliary matrices: \nX = {X}, \nY = {Y}')

# calculating vector B = transposed(b0, b1, .... bk)

B = multiply(multiply(inverse(multiply(transposed(X), X)), transposed(X)), Y)

print(f'\nequation of hyperplane is: y = {B[0][0]} +', end=' ')

for i in range(1, k):
    print(f'{round(B[i][0], DECIMAL_PRECISION)} * x{i} +', end=' ')

print(f'{round(B[k][0], DECIMAL_PRECISION)} * x{k}')




