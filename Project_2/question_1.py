import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.typing import NDArray
from scipy.linalg import lu


def cramer(A: NDArray[np.float64], b: NDArray[np.float64]):
    n = len(b)
    detA = np.linalg.det(A)
    if np.isclose(detA, 0):
        raise np.linalg.LinAlgError("O sistema não possui solução única (detA = 0).")

    x = np.zeros(n, dtype=float)
    n_flops_cramer = (n + 1) * (2 / 3 * n ** 3 + n ** 2)

    for i in range(n):
        Ai = np.copy(A)
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / detA

    return x, n_flops_cramer


def gauss_elimination(A: NDArray[np.float64], b: NDArray[np.float64]):
    if A.shape[0] != A.shape[1] or len(b) != A.shape[0]:
        raise ValueError("Dimensões incompatíveis.")

    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1)])
    n_flops = (2 / 3 * n ** 3) + (n ** 2)
    for i in range(n):
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if np.isclose(Ab[max_row, i], 0.0):
            raise np.linalg.LinAlgError("O sistema é singular (pivô nulo)")
        Ab[[i, max_row]] = Ab[[max_row, i]]
        Ab[i] = Ab[i] / Ab[i, i]
        for j in range(i + 1, n):
            Ab[j] -= Ab[i] * Ab[j, i]

    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.sum(Ab[i, i + 1:n] * x[i + 1:n])

    return x, n_flops


def lu_decomposition(A: NDArray[np.float64], b: NDArray[np.float64]):
    if A.shape[0] != A.shape[1] or len(b) != A.shape[0]:
        raise ValueError("Dimensões incompatíveis.")

    n = len(b)
    n_flops = (2 / 3 * n ** 3) + (2 * n ** 2)

    P, L, U = lu(A)
    y = np.linalg.solve(L, P @ b)
    x = np.linalg.solve(U, y)

    return x, n_flops


def jacobi(
        A: NDArray[np.float64],
        b: NDArray[np.float64],
        tol: float = 1e-10,
        max_iter: int = 1000
):
    D = np.diag(A)
    if np.any(D == 0):
        raise ValueError("A matriz A possui elementos nulos na diagonal.")

    n = len(b)
    x = np.zeros(n)
    R = A - np.diagflat(D)

    for k in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x) < tol:
            return x_new, k + 1
        x = x_new

    raise np.linalg.LinAlgError("O método de Jacobi não convergiu.")


def gauss_seidel(
        A: NDArray[np.float64],
        b: NDArray[np.float64],
        tol: float = 1e-10,
        max_iter: int = 1000
):
    if np.any(np.diag(A) == 0):
        raise ValueError("A matriz A possui elementos nulos na diagonal.")

    n = len(b)
    x = np.zeros_like(b, dtype=float)

    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sum_known = np.dot(A[i, :i], x_new[:i])
            sum_old = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum_known - sum_old) / A[i, i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new, k + 1
        x = x_new

    raise np.linalg.LinAlgError("O método de Gauss-Seidel não convergiu.")


def estabilidade_numerica(A):
    return np.linalg.cond(A)


# ----------------- EXECUÇÃO DA COMPARAÇÃO REFINADA -----------------

# Usando a matriz de 10x10 que é diagonalmente dominante (boa para iterativos)
A_ref = np.array([
    [133.362, -1.222, 7.172, 3.947, -8.116, 9.512, 5.223, 5.721, -7.438, -0.992],
    [-2.584, 103.762, 2.877, 6.455, -1.132, -5.455, 1.092, -8.724, 6.553, 2.633],
    [5.162, -2.909, 129.933, 7.862, 5.568, -6.107, -0.666, -9.124, -6.914, 3.661],
    [4.895, 9.35, -3.483, 113.235, -0.609, -6.211, -7.402, -0.486, -5.462, 3.396],
    [-1.257, 6.654, 4.005, -3.753, 107.756, 6.095, -2.25, -4.233, 3.65, -7.205],
    [-6.002, -9.853, 5.738, 3.297, 4.103, 125.258, -0.822, 1.375, -7.204, -7.709],
    [3.368, -0.578, 1.305, 5.3, 2.694, 1.072, 82.218, -3.921, -9.384, -1.266],
    [-5.708, -1.829, 7.068, -5.321, -8.834, -4.372, -4.128, 120.2, 1.141, 5.678],
    [3.286, -1.872, 6.28, -6.661, -9.546, -8.199, 4.447, -0.762, 112.687, 0.021],
    [-6.954, 3.926, -1.077, -2.38, -3.97, 2.606, -2.764, -8.247, -7.64, 108.907]
], dtype=float)
b_ref = np.array(
    [81.716, 39.941, -46.826, 93.835, 55.75, 43.378, -10.128, -45.552, -80.722, 80.52],
    dtype=float)

n_dim = A_ref.shape[0]
x_real = np.linalg.solve(A_ref, b_ref)
cond_A = estabilidade_numerica(A_ref)

print(f"Dimensão do Sistema: n = {n_dim}")
print(f"Número de Condicionamento ($\kappa(A)$): {cond_A:.2e} (Estabilidade do Problema)")

# Dicionários para armazenar resultados
tempos = {}
erros = {}
metricas_custo = {}  # FLOPs (Diretos) ou Iterações (Iterativos)
categorias = {}

metodos_ref = {
    # Cramer é mantido para n=10, mas espera-se um FLOPs alto
    'Cramer': (cramer, A_ref, b_ref, 'Direto'),
    'Eliminação de Gauss': (gauss_elimination, A_ref, b_ref, 'Direto'),
    'Decomposição LU': (lu_decomposition, A_ref, b_ref, 'Direto'),
    'Jacobi': (jacobi, A_ref, b_ref, 'Iterativo'),
    'Gauss-Seidel': (gauss_seidel, A_ref, b_ref, 'Iterativo')
}

for nome, (func, A, b, categoria) in metodos_ref.items():
    try:
        inicio = time.time()
        x_calc, n_cost = func(A, b)
        fim = time.time()

        erro_norma = np.linalg.norm(x_calc - x_real)

        tempos[nome] = fim - inicio
        erros[nome] = erro_norma
        metricas_custo[nome] = n_cost
        categorias[nome] = categoria

    except Exception as e:
        print(f"Erro ao executar {nome}: {e}")

# Separação dos dados
dados_diretos = {
    'nomes': [n for n in metodos_ref if categorias.get(n) == 'Direto' and n in tempos],
    'tempos': [tempos[n] for n in metodos_ref if
               categorias.get(n) == 'Direto' and n in tempos],
    'erros': [erros[n] for n in metodos_ref if
              categorias.get(n) == 'Direto' and n in tempos],
    'custo': [metricas_custo[n] for n in metodos_ref if
              categorias.get(n) == 'Direto' and n in tempos],
}

dados_iterativos = {
    'nomes': [n for n in metodos_ref if categorias.get(n) == 'Iterativo' and n in tempos],
    'tempos': [tempos[n] for n in metodos_ref if
               categorias.get(n) == 'Iterativo' and n in tempos],
    'erros': [erros[n] for n in metodos_ref if
              categorias.get(n) == 'Iterativo' and n in tempos],
    'custo': [metricas_custo[n] for n in metodos_ref if
              categorias.get(n) == 'Iterativo' and n in tempos],
}


# ----------------- Geração dos Gráficos Separados -----------------

def plot_comparacao(dados, titulo_sufixo, custo_label, color):
    """Função auxiliar para plotar os 3 gráficos para um grupo de métodos."""

    nomes = dados['nomes']

    if not nomes:
        print(f"Nenhum dado disponível para {titulo_sufixo}.")
        return

    plt.figure(figsize=(18, 5))
    plt.suptitle(f'Comparação de Métodos {titulo_sufixo} (n={n_dim})', fontsize=16)

    # 1. Tempo de Execução
    plt.subplot(1, 3, 1)
    plt.bar(nomes, dados['tempos'], color=color)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.title('Tempo de Execução (s) - Escala Logarítmica')
    plt.ylabel('Tempo (s)')

    # 2. Erro Numérico
    plt.subplot(1, 3, 2)
    plt.bar(nomes, dados['erros'], color=color)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.title('Erro Numérico (Norma Euclidiana) - Escala Logarítmica')
    plt.ylabel(r'$\left\|\mathbf{x} - \mathbf{x}_{\text{ref}}\right\|_2$')

    # 3. Métrica de Custo (FLOPs ou Iterações)
    plt.subplot(1, 3, 3)
    plt.bar(nomes, dados['custo'], color=color)
    plt.xticks(rotation=45, ha='right')
    if custo_label == 'Iterações (k)':
        # Para Iterações, usar escala linear
        plt.title(f'Custo Computacional: {custo_label}')
    else:
        # Para FLOPs, usar escala logarítmica devido à grande magnitude
        plt.yscale('log')
        plt.title(f'Custo Computacional: {custo_label} (Log Scale)')

    plt.ylabel(custo_label)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Plotar Métodos Diretos
plot_comparacao(dados_diretos,
                'Diretos (Baseados em Fatoração/Transformação)',
                'FLOPs (Magnitude)',
                'skyblue')

# Plotar Métodos Iterativos
plot_comparacao(dados_iterativos,
                'Iterativos (Baseados em Aproximações Sucedidas)',
                'Iterações (k)',
                'blue')