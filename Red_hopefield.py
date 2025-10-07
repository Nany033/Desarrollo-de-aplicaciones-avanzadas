def bipolarize(mat):
    v = []
    for fila in mat:
        for x in fila:
            v.append(1 if x == 1 else -1)
    return v

def reshape(vec, filas, cols):
    mat, k = [], 0
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
    for i in range(len(M)):
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

# --- Red Hopfield ---

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

# --- Lectura de patrones desde archivo TXT ---

def cargar_patrones(nombre_archivo):
    patrones = {}
    with open(nombre_archivo, 'r') as f:
        lineas = f.readlines()

    nombre = None
    figura = []
    for linea in lineas:
        linea = linea.strip()
        if not linea:
            if nombre and figura:
                patrones[nombre] = [[int(c) for c in fila] for fila in figura]
                figura = []
            continue
        if linea.startswith("#"):
            nombre = linea[1:].strip()
        else:
            figura.append(linea)
    if nombre and figura:
        patrones[nombre] = [[int(c) for c in fila] for fila in figura]
    return patrones

# --- DEMO PRINCIPAL ---

if __name__ == "__main__":
    # Cargar figuras desde archivo
    archivo = "patrones.txt"
    patrones = cargar_patrones(archivo)

    # Crear red 8x5
    net = Hopfield(8, 5)
    net.train(list(patrones.values()))

    # Mostrar los patrones cargados
    print("Patrones cargados desde archivo:\n")
    for nombre, figura in patrones.items():
        print(f"Figura: {nombre}")
        for fila in figura:
            print(''.join(['#' if x==1 else '.' for x in fila]))
        print()

    # Probar con ruido en '?'
    noisy = [fila[:] for fila in patrones["QUESTION"]]
    noisy[1][1] = 1
    noisy[2][3] = 1

    print("Patrón ruidoso (?):")
    for fila in noisy:
        print(''.join(['#' if x==1 else '.' for x in fila]))

    result = net.recall(noisy)

    print("\nResultado tras recuperación:")
    for fila in result:
        print(''.join(['#' if x==1 else '.' for x in fila]))
