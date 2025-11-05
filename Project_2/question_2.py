import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib.animation import FuncAnimation as funani

#Temperaturas 16/10 min:16 max:30; 17/10 min:17 max:33; 18/10 min:16 max:24; 19/10 min:12 max:15; 20/10 min:11 max:16; 21/10 min:11 max:19; 22/10 min:11 max:22; 23/10 min:12 max:23
#24/10 min:13 max:25; 25/10 min:15 max:30

#Funciona bem pra interpolação, porem não funciona em extrapolação, valores muito absurdos

#P(x) exato = -0.00027419532627866*x**9 + 0.0511656746031745*x**8 - 4.23274636243387*x**7 + 203.736458333333*x**6 - 6287.72624421297*x**5 + 129023.222395833*x**4 - 1760222.08279872*x**3 + 15394778.4899801*x**2 - 78318294.4579366*x + 176568315.0

dias=np.array([16,17,18,19,20,21,22,23,24,25])
tempm=np.array([23,25,20,13.5,13.5,15,16.5,17.5,19,22.5])
xplt=np.linspace(dias[0],dias[-1])
ylag= []
ynew = []
ygnew = []
#Classe tipo de interpolador
class interpoli:
    def __init__(self,x,y):
        self.x = np.array(x)    
        self.y = np.array(y)
        self.n = len(x)
        self.h = x[1] - x[0]
    def Lagrange(self, k):
       ylag=0.0
       for xi,yj in zip( self.x,self.y):
           ylag+=yj*np.prod((k-self.x[self.x!=xi])/(xi-self.x[self.x!=xi]))
       return ylag
    def Newton(self,k):
        ynew=np.zeros((self.n,self.n))
        ynew[:,0]=self.y
        for j in range(1,self.n):
            for i in range(self.n-j):
                ynew[i,j]=(ynew[i+1,j-1]-ynew[i,j-1])/(self.x[i+j]-self.x[i])
        resultado = ynew[0, 0]  
        produto = 1.0
        
        for i in range(1, self.n):
            produto *= (k - self.x[i-1])
            resultado += ynew[0, i] * produto
        
        return resultado
    def GregoryNewton(self, k):
        ydif = np.zeros((self.n, self.n))
        ydif[:,0] = self.y
        
        for j in range(1, self.n):
            for i in range(self.n - j):
                ydif[i,j] = ydif[i+1,j-1] - ydif[i,j-1]
        
        s = (k - self.x[0]) / self.h
        ygn = ydif[0,0]
        produto = 1.0
        
        for j in range(1, self.n):
            produto *= (s - (j-1))
            ygn += (produto * ydif[0,j]) / math.factorial(j)
        
        return ygn
    def erro_teorico(self, k, grau):
        M = 10.0
        pi_x = 1.0
        for i in range(grau + 1):
            pi_x *= abs(k - self.x[i])
        return (M / math.factorial(grau + 1)) * pi_x
    
    def erro_pratico(self, k, grau):
        if grau == self.n - 1:
            return 0
        pontos_x = self.x[:grau+1]
        pontos_y = self.y[:grau+1]
        P_grau = 0.0
        for xi, yj in zip(pontos_x, pontos_y):
            P_grau += yj * np.prod((k - pontos_x[pontos_x != xi]) / (xi - pontos_x[pontos_x != xi]))
        P_max = self.Lagrange(k)
        
        return abs(P_max - P_grau)

interpolador = interpoli(dias, tempm)       
ylag = [interpolador.Lagrange(x) for x in xplt]
ynew = [interpolador.Newton(x) for x in xplt]
ygnew = [interpolador.GregoryNewton(x) for x in xplt]
erros_lagrange = []
erros_newton = []
erros_gregory = []

for i, (dia, temp_esperada) in enumerate(zip(dias, tempm)):
    temp_lag = interpolador.Lagrange(dia)
    temp_new = interpolador.Newton(dia)
    temp_greg = interpolador.GregoryNewton(dia)
    
    erro_lag = abs(temp_lag - temp_esperada)
    erro_new = abs(temp_new - temp_esperada)
    erro_greg = abs(temp_greg - temp_esperada)
    
    erros_lagrange.append(erro_lag)
    erros_newton.append(erro_new)
    erros_gregory.append(erro_greg)
    
    print(f"Dia {dia}: Esperado = {temp_esperada}°C")
    print(f"  Lagrange: {temp_lag:.10f}°C | Erro: {erro_lag:.10e}")
    print(f"  Newton:   {temp_new:.10f}°C | Erro: {erro_new:.10e}")
    print(f"  Gregory:  {temp_greg:.10f}°C | Erro: {erro_greg:.10e}\n")

print("ANÁLISE DE ERRO POR GRAU DO POLINÔMIO\n")

pontos_teste = [16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5]
graus = list(range(1, len(dias)))
erros_teoricos_por_grau = []
erros_praticos_por_grau = []

for grau in graus:
    erros_teoricos = []
    erros_praticos = []
    
    for x_teste in pontos_teste:
        erro_teorico = interpolador.erro_teorico(x_teste, grau)
        erro_pratico = interpolador.erro_pratico(x_teste, grau)
        
        erros_teoricos.append(erro_teorico)
        erros_praticos.append(erro_pratico)
    
    erros_teoricos_por_grau.append(max(erros_teoricos))
    erros_praticos_por_grau.append(max(erros_praticos))
    
    print(f"Grau {grau}: Erro Teórico Máx = {max(erros_teoricos):.2e} | Erro Prático Máx = {max(erros_praticos):.2e}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(graus, erros_teoricos_por_grau, 'ro-', linewidth=2, markersize=6, label='Erro Teórico')
plt.plot(graus, erros_praticos_por_grau, 'bo-', linewidth=2, markersize=6, label='Erro Prático')
plt.xlabel('Grau do Polinômio')
plt.ylabel('Erro Máximo (°C)')
plt.title('Evolução do Erro com o Grau')
plt.legend()
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.semilogy(graus, erros_teoricos_por_grau, 'ro-', linewidth=2, markersize=6, label='Erro Teórico')
plt.semilogy(graus, erros_praticos_por_grau, 'bo-', linewidth=2, markersize=6, label='Erro Prático')
plt.xlabel('Grau do Polinômio')
plt.ylabel('Erro Máximo (escala log)')
plt.title('Evolução do Erro - Escala Logarítmica')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

    
fig, ax = plt.subplots(figsize=(12, 7))
def animate(frame):
    ax.clear()
    ax.plot(dias, tempm, 'ko', markersize=8, label='Dados originais', markerfacecolor='red')
    if frame >= 1:
        progresso_lag = min(frame, len(xplt))
        ax.plot(xplt[:progresso_lag], ylag[:progresso_lag], 'b-', linewidth=3,  label='Lagrange', alpha=0.8)
    if frame >= 51:
        progresso_new = min(frame - 50, len(xplt))
        ax.plot(xplt[:progresso_new], ynew[:progresso_new], 'r-', linewidth=4, label='Newton', alpha=0.8)
    if frame >= 101:
        progresso_greg = min(frame - 100, len(xplt))
        ax.plot(xplt[:progresso_greg], ygnew[:progresso_greg], 'g-', linewidth=5, label='Gregory-Newton', alpha=0.8)
    
    ax.set_xlim(dias[0] - 0.5, dias[-1] + 0.5)
    ax.set_ylim(10, 30)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Dia do Mês')
    ax.set_ylabel('Temperatura (°C)')
    
    if frame < 50:
        ax.set_title(f'Interpolação-Lagrange')
    elif frame < 10:
        ax.set_title(f'Interpolação-Newton')
    elif frame < 150:
        ax.set_title(f'Interpolação-Gregory-Newton')
    else:
        ax.set_title('Todas as Interpolações')
    
    ax.legend()
    return ax,
total_frames = 200
anim = funani(fig, animate, frames=total_frames, interval=50, blit=False, repeat=True)
anim.save('Grafico-Questão 2.gif', writer='pillow', fps=30)
plt.show()
