def bipolarize(mat):
    """Convierte una matriz 0/1 a vector bipolar (-1, 1)."""
    v = []
    for fila in mat:
        for x in fila:
            v.append(1 if x == 1 else -1)
    return v

def reshape(vec, filas, cols):
    """Convierte vector a matriz filas x cols."""
    mat = []
    k = 0
    for i in range(filas):
        fila = []
        for j in range(cols):
            fila.append(vec[k])
            k += 1
        mat.append(fila)
    return mat

def outer_product(v):
    n = len(v)
    M = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = v[i] * v[j]
    return M

def add_matrices(A, B):
    n = len(A)
    M = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = A[i][j] + B[i][j]
    return M

def zero_diagonal(M):
    n = len(M)
    for i in range(n):
        M[i][i] = 0
    return M

def matrix_vector_mul(M, v):
    n = len(M)
    res = [0 for _ in range(n)]
    for i in range(n):
        s = 0
        for j in range(n):
            s += M[i][j] * v[j]
        res[i] = s
    return res

def activation(v):
    return [1 if x >= 0 else -1 for x in v]



class Hopfield:
    def __init__(self, filas, cols):
        self.filas = filas
        self.cols = cols
        self.n = filas * cols
        self.W = [[0 for _ in range(self.n)] for _ in range(self.n)]

    def train(self, patterns):
        for p in patterns:
            v = bipolarize(p)
            op = outer_product(v)
            self.W = add_matrices(self.W, op)
        self.W = zero_diagonal(self.W)

    def recall(self, pattern, max_iter=10):
        state = bipolarize(pattern)
        for _ in range(max_iter):
            new_state = activation(matrix_vector_mul(self.W, state))
            if new_state == state:
                break
            state = new_state
        return reshape(state, self.filas, self.cols)



if __name__ == "__main__":

    #matrices hechas con ayuda de IA (chatGPT)
    STAR = [
        [0,0,1,0,0],
        [0,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ]

    SLASH = [
        [0,0,0,0,1],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,0],
        [1,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ]

    QUESTION = [
        [0,1,1,1,0],
        [1,0,0,0,1],
        [0,0,0,0,1],
        [0,0,0,1,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0]
    ]

    PLUS = [
        [0,0,1,0,0],
        [0,0,1,0,0],
        [1,1,1,1,1],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ]

    net = Hopfield(8, 5)
    net.train([STAR, SLASH, QUESTION, PLUS])

    noisy = [fila[:] for fila in QUESTION]
    noisy[1][1] = 1
    noisy[2][2] = 1

    print("Patrón original (?):")
    for fila in QUESTION:
        print(''.join(['#' if x==1 else '.' for x in fila]))

    print("\nPatrón ruidoso:")
    for fila in noisy:
        print(''.join(['#' if x==1 else '.' for x in fila]))

    result = net.recall(noisy)

    print("\nResultado de la red:")
    for fila in result:
        print(''.join(['#' if x==1 else '.' for x in fila]))
