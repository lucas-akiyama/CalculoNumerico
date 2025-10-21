import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.typing import NDArray
from scipy.linalg import lu

# --- Funções para os métodos ---

def cramer(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Resolve um sistema linear de equações utilizando a Regra de Cramer.

    Este método calcula as soluções do sistema A·x = b, onde A é uma matriz
    quadrada e b é o vetor de termos independentes. Ele substitui cada coluna
    de A pelo vetor b, calcula o determinante e aplica a fórmula de Cramer
    para encontrar as incógnitas.

    Args:
        A (numpy.ndarray): Matriz dos coeficientes do sistema (n x n).
        b (numpy.ndarray): Vetor dos termos independentes (n,).

    Returns:
        numpy.ndarray: Vetor solução x do sistema (n,), contendo os valores
        das incógnitas.

    Raises:
        numpy.linalg.LinAlgError: Se o determinante de A for zero, indicando
        que o sistema não possui solução única.
    """
    n = len(b)
    detA = np.linalg.det(A)
    if np.isclose(detA, 0, 0):
        raise np.linalg.LinAlgError("O sistema não possui solução única. (determinante = 0)")
    x = np.zeros(n, dtype=float)
    for i in range(n):
        Ai = np.copy(A)
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / detA
    return x

def gauss_elimination(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Resolve um sistema linear de equações utilizando o método de eliminação de Gauss
    (com pivoteamento parcial).

    O algoritmo transforma a matriz aumentada [A|b] em uma forma triangular superior
    por meio de operações elementares nas linhas. Em seguida, aplica retro-substituição
    para determinar o vetor solução x.

    Args:
        A (numpy.ndarray): Matriz dos coeficientes do sistema (n x n).
        b (numpy.ndarray): Vetor dos termos independentes (n,).

    Returns:
        numpy.ndarray: Vetor solução x do sistema (n,), contendo os valores
        das incógnitas.

    Raises:
        ValueError: Se A não for uma matriz quadrada ou se b não tiver
        dimensão compatível.
        numpy.linalg.LinAlgError: Se o sistema for singular (possuir
        pivô igual a zero durante o processo).
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada.")
    if len(b) != A.shape[0]:
        raise ValueError("O vetor b deve ter o mesmo número de linhas que A")

    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1)])
    for i in range(n):
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if np.isclose(Ab[max_row, i], 0.0):
            raise np.linalg.LinAlgError("O sistema é singular (pivô nulo)")
        Ab[[i, max_row]] = Ab[[max_row, i]]
        Ab[i] = Ab[i] / Ab[i, i]
        for j in range(i+1, n):
            Ab[j] -= Ab[i] * Ab[j, i]

    x = np.zeros(n, dtype=float)
    for i in range(n-1, -1, -1):
        x[i] = Ab[i, -1] - np.sum(Ab[i, i+1:n] * x[i+1:n])
    return x

def lu_decomposition(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Resolve um sistema linear de equações A·x = b utilizando a Decomposição LU.

    Args:
        A (numpy.ndarray): Matriz dos coeficientes do sistema (n x n).
        b (numpy.ndarray): Vetor dos termos independentes (n,).

    Returns:
        numpy.ndarray: Vetor solução x do sistema (n,), contendo os valores
        das incógnitas.

    Raises:
        ValueError: Se A não for uma matriz quadrada ou se b não tiver
        dimensão compatível.
        numpy.linalg.LinAlgError: Se o sistema for singular (determinante nulo).
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada.")
    if len(b) != A.shape[0]:
        raise ValueError("O vetor b deve ter o mesmo número de linhas que A.")
    if np.isclose(np.linalg.det(A), 0.0):
        raise np.linalg.LinAlgError("O sistema é singular (determinante nulo).")
    P, L, U = lu(A)
    y = np.linalg.solve(L, P @ b)
    x = np.linalg.solve(U, y)
    return x

def jacobi(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    tol: float = 1e-10,
    max_iter: int = 1000
) -> NDArray[np.float64]:
    """
    Resolve o sistema linear A·x = b utilizando o método iterativo de Jacobi.

    O método de Jacobi é um algoritmo iterativo para resolver sistemas lineares
    diagonais dominantes. Ele parte de uma estimativa inicial (geralmente x = 0)
    e atualiza cada componente de x de forma independente em cada iteração.

    A convergência é garantida se a matriz A for **diagonalmente dominante** ou
    **simétrica definida positiva**.

    Args:
        A (numpy.ndarray): Matriz dos coeficientes do sistema (n x n).
        b (numpy.ndarray): Vetor dos termos independentes (n,).
        tol (float, opcional): Tolerância para o critério de parada.
            Iterações param quando ||x_new - x|| < tol. Padrão: 1e-10.
        max_iter (int, opcional): Número máximo de iterações permitidas.
            Padrão: 1000.

    Returns:
        numpy.ndarray: Vetor solução aproximada x do sistema (n,).

    Raises:
        ValueError: Se A não for uma matriz quadrada, se b tiver tamanho incompatível
        ou se algum elemento da diagonal de A for zero.
        np.linalg.LinAlgError: Se o método não convergir dentro do limite de iterações.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada.")
    if len(b) != A.shape[0]:
        raise ValueError("O vetor b deve ter o mesmo número de linhas que A.")
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError(
            "A matriz A possui elementos nulos na diagonal (divisão por zero).")

    n = len(b)
    x = np.zeros(n)
    R = A - np.diagflat(D)
    for _ in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    raise np.linalg.LinAlgError("O método de Jacobi não convergiu dentro do número máximo de iterações.")

def gauss_seidel(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
    tol: float = 1e-10,
    max_iter: int = 1000
) -> NDArray[np.float64]:
    """
    Resolve o sistema linear A·x = b utilizando o método iterativo de Gauss-Seidel.

    O método de Gauss-Seidel é uma variação do método de Jacobi, onde as atualizações
    de cada componente de x são imediatamente utilizadas nas próximas iterações da mesma
    passagem. Isso geralmente melhora a velocidade de convergência em relação ao Jacobi.

    A convergência é garantida se a matriz A for **diagonalmente dominante** ou
    **simétrica definida positiva**.

    Args:
        A (numpy.ndarray): Matriz dos coeficientes do sistema (n x n).
        b (numpy.ndarray): Vetor dos termos independentes (n,).
        tol (float, opcional): Tolerância para o critério de parada.
            Iterações param quando ||x_new - x|| < tol. Padrão: 1e-10.
        max_iter (int, opcional): Número máximo de iterações permitidas.
            Padrão: 1000.

    Returns:
        numpy.ndarray: Vetor solução aproximada x do sistema (n,).

    Raises:
        ValueError: Se A não for quadrada, se b tiver tamanho incompatível
            ou se houver elementos nulos na diagonal de A.
        np.linalg.LinAlgError: Se o método não convergir dentro do número máximo de iterações.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada.")
    if len(b) != A.shape[0]:
        raise ValueError("O vetor b deve ter o mesmo número de linhas que A.")
    if np.any(np.diag(A) == 0):
        raise ValueError(
            "A matriz A possui elementos nulos na diagonal (divisão por zero).")

    n = len(b)
    x = np.zeros_like(b, dtype=float)
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new

    raise np.linalg.LinAlgError(
        "O método de Gauss-Seidel não convergiu dentro do número máximo de iterações.")

# --- Teste com sistema pequeno (3x3) ---
A_small = np.array([[3, -1, 1], [-2, 4, 0], [1, 1, 5]], dtype=float)
b_small = np.array([5, 6, -4], dtype=float)

# --- Teste com sistema grande (10x10) ---
A_large = (np.array([
    [13.3362, -0.1222, 0.7172, 0.3947, -0.8116, 0.9512, 0.5223, 0.5721, -0.7438, -0.0992],
    [-0.2584, 10.3762, 0.2877, 0.6455, -0.1132, -0.5455, 0.1092, -0.8724, 0.6553, 0.2633],
    [ 0.5162, -0.2909, 12.9933, 0.7862, 0.5568, -0.6107, -0.0666, -0.9124, -0.6914, 0.3661],
    [ 0.4895, 0.935 , -0.3483, 11.3235, -0.0609, -0.6211, -0.7402, -0.0486, -0.5462, 0.3396],
    [-0.1257, 0.6654, 0.4005, -0.3753, 10.7756, 0.6095, -0.225 , -0.4233, 0.365 , -0.7205],
    [-0.6002, -0.9853, 0.5738, 0.3297, 0.4103, 12.5258, -0.0822, 0.1375, -0.7204, -0.7709],
    [ 0.3368, -0.0578, 0.1305, 0.53 , 0.2694, 0.1072, 8.2218, -0.3921, -0.9384, -0.1266],
    [-0.5708, -0.1829, 0.7068, -0.5321, -0.8834, -0.4372, -0.4128, 12.02 , 0.1141, 0.5678],
    [ 0.3286, -0.1872, 0.628 , -0.6661, -0.9546, -0.8199, 0.4447, -0.0762, 11.2687, 0.0021],
    [-0.6954, 0.3926, -0.1077, -0.238 , -0.397 , 0.2606, -0.2764, -0.8247, -0.764 , 10.8907]
]) * 10)
b_large = np.array([ 8.1716, 3.9941, -4.6826, 9.3835, 5.575 , 4.3378, -1.0128, -4.5552, -8.0722, 8.052 ]) * 10

# --- Avaliar tempo e erro ---
metodos = {
    'Cramer (3x3)': (cramer, A_small, b_small),
    'Gauss': (gauss_elimination, A_small, b_small),
    'LU': (lu_decomposition, A_small, b_small),
    'Jacobi (10x10)': (jacobi, A_large, b_large),
    'Gauss-Seidel (10x10)': (gauss_seidel, A_large, b_large)
}

tempos = []
erros = []

for nome, (func, A, b) in metodos.items():
    inicio = time.time()
    x = func(A, b)
    fim = time.time()
    x_real = np.linalg.solve(A, b)
    erro = np.linalg.norm(x - x_real)
    tempos.append(fim - inicio)
    erros.append(erro)

# --- Gráfico comparativo ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(metodos.keys(), tempos)
plt.xticks(rotation=45, ha='right')
plt.title('Tempo de Execução (s)')
plt.subplot(1, 2, 2)
plt.bar(metodos.keys(), erros)
plt.xticks(rotation=45, ha='right')
plt.title('Erro Numérico (norma)')
plt.tight_layout()
plt.show()
