import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

def f(x):
    
    return np.sin(5*x)*np.exp(-2)+np.cos(x)

def metodo_simpson_38(f, a, b, n):
    
    if n % 3 != 0:
        n = (n // 3) * 3  
    
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n

    integral = (3 * h / 8) * (y[0] + y[n] + 
                             3 * np.sum(y[1:n:3] + y[2:n:3]) + 
                             2 * np.sum(y[3:n-1:3]))
    return integral

def criar_animacao_comparacao_simples(f, a, b, n_max=60):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    
    x_fino = np.linspace(a, b, 1000)
    y_fino = f(x_fino)
    
    
    valor_referencia, _ = spi.quad(f, a, b)
    
    
    ax4.plot(x_fino, y_fino, 'b-', linewidth=2, label='f(x)')
    ax4.fill_between(x_fino, y_fino, alpha=0.3, color='blue', label='Área sob a curva')
    ax4.set_xlim(a, b)
    ax4.set_ylim(min(y_fino) - 0.2, max(y_fino) + 0.2)
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.set_title('Função Original\nf(x) = sin(5x)exp(-2) + cos(x)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(a, b)
        ax.set_ylim(min(y_fino) - 0.2, max(y_fino) + 0.2)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.3)
    
    ax1.set_title('Método dos Trapézios')
    ax2.set_title('Método de Simpson 1/3')
    ax3.set_title('Método de Simpson 3/8')
    
    
    # Trapézios
    linha_trap, = ax1.plot([], [], 'b-', linewidth=1)
    pontos_trap, = ax1.plot([], [], 'ro', markersize=3)
    
    # Simpson 1/3
    linha_simp, = ax2.plot([], [], 'b-', linewidth=1)
    pontos_simp, = ax2.plot([], [], 'go', markersize=3)
    
    # Simpson 3/8
    linha_simp38, = ax3.plot([], [], 'b-', linewidth=1)
    pontos_simp38, = ax3.plot([], [], 'bo', markersize=3)
    

    n_valores = list(range(6, n_max + 1, 6))  
    
    def init():
        linha_trap.set_data([], [])
        pontos_trap.set_data([], [])
        linha_simp.set_data([], [])
        pontos_simp.set_data([], [])
        linha_simp38.set_data([], [])
        pontos_simp38.set_data([], [])
        return linha_trap, pontos_trap, linha_simp, pontos_simp, linha_simp38, pontos_simp38
    
    def animate(n):
        # Limpar áreas anteriores
        for ax in [ax1, ax2, ax3]:
            for patch in ax.patches:
                patch.remove()
            for line in ax.lines[2:]:  # Manter apenas as duas primeiras linhas
                if line not in [linha_trap, pontos_trap, linha_simp, pontos_simp, linha_simp38, pontos_simp38]:
                    line.remove()
        
        
        x = np.linspace(a, b, n + 1)
        y = f(x)
        
        # MÉTODO 1: TRAPÉZIOS
        linha_trap.set_data(x, y)
        pontos_trap.set_data(x, y)
        
        
        for i in range(n):
            vertices = [(x[i], 0), (x[i], y[i]), (x[i+1], y[i+1]), (x[i+1], 0)]
            polygon = Polygon(vertices, closed=True, alpha=0.3, color='red')
            ax1.add_patch(polygon)
        
        
        integral_trap = spi.trapezoid(y, x)
        erro_trap = abs(integral_trap - valor_referencia)
        ax1.set_title(f'Método dos Trapézios\nn = {n}, I = {integral_trap:.6f}\nErro = {erro_trap:.2e}')
        
        # MÉTODO 2: SIMPSON 1/3
        linha_simp.set_data(x, y)
        pontos_simp.set_data(x, y)
        
        
        if n >= 4 and n % 2 == 0:
            for i in range(0, n-1, 2):
                x_parab = np.linspace(x[i], x[i+2], 100)
                coef = np.polyfit(x[i:i+3], y[i:i+3], 2)
                y_parab = np.polyval(coef, x_parab)
                ax2.plot(x_parab, y_parab, 'r--', alpha=0.7, linewidth=1)
                ax2.fill_between(x_parab, y_parab, alpha=0.2, color='green')
        
        
        integral_simp = spi.simpson(y, x) if n % 2 == 0 else integral_trap
        erro_simp = abs(integral_simp - valor_referencia)
        ax2.set_title(f'Método de Simpson 1/3\nn = {n}, I = {integral_simp:.6f}\nErro = {erro_simp:.2e}')
        
        # MÉTODO 3: SIMPSON 3/8
        linha_simp38.set_data(x, y)
        pontos_simp38.set_data(x, y)
        
        if n >= 3 and n % 3 == 0:
            for i in range(0, n-2, 3):
                x_cubic = np.linspace(x[i], x[i+3], 100)
                coef = np.polyfit(x[i:i+4], y[i:i+4], 3)
                y_cubic = np.polyval(coef, x_cubic)
                ax3.plot(x_cubic, y_cubic, 'r--', alpha=0.7, linewidth=1)
                ax3.fill_between(x_cubic, y_cubic, alpha=0.2, color='purple')
        
        
        integral_simp38 = metodo_simpson_38(f, a, b, n) if n % 3 == 0 else integral_trap
        erro_simp38 = abs(integral_simp38 - valor_referencia)
        ax3.set_title(f'Método de Simpson 3/8\n n = {n}, I = {integral_simp38:.6f}\nErro = {erro_simp38:.2e}')
        
        return linha_trap, pontos_trap, linha_simp, pontos_simp, linha_simp38, pontos_simp38
    
    anim = FuncAnimation(fig, animate, frames=n_valores, init_func=init, 
                        interval=800, repeat=True, blit=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim

# EXECUÇÃO PRINCIPAL
a, b = 0, 3 * np.pi
    
anim = criar_animacao_comparacao_simples(f, a, b, n_max=60)

# anim.save('comparacao_integracao.gif', writer='pillow', fps=2)
