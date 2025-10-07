def bipolarize(vec):
    return [1 if v == 1 else -1 for v in vec]

def outer_product(vec):
    n = len(vec)
    M = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = vec[i] * vec[j]
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
        suma = 0
        for j in range(n):
            suma += M[i][j] * v[j]
        res[i] = suma
    return res

def activation(v):
    return [1 if x >= 0 else -1 for x in v]

def hopfield_train(patterns):
    n = len(patterns[0])
    W = [[0 for _ in range(n)] for _ in range(n)]
    for p in patterns:
        M = outer_product(p)
        W = add_matrices(W, M)
    W = zero_diagonal(W)
    return W

def hopfield_recall(W, input_vec, max_iter=10):
    state = input_vec[:]
    for _ in range(max_iter):
        new_state = activation(matrix_vector_mul(W, state))
        if new_state == state:
            break  
        state = new_state
    return state

if __name__ == "__main__":
    x1 = [1, 1, 1, 0]
    x2 = [0, 0, 0, 1]
    x1_b = bipolarize(x1)
    x2_b = bipolarize(x2)
    print("x1 bipolar:", x1_b)
    print("x2 bipolar:", x2_b)
    W = hopfield_train([x1_b, x2_b])
    print("\nMatriz de pesos:")
    for fila in W:
        print(fila)
    entrada = x1_b[:]  
    salida = hopfield_recall(W, entrada)

    print("\nEntrada:", entrada)
    print("Salida :", salida)
    if salida == x1_b:
        print("Patrón x1 encontrado")
    elif salida == x2_b:
        print("Patrón x2 encontrado")
    else:
        print("No converge a un patrón conocido")

