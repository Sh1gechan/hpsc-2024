import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

NX = 41
NY = 41

def main():
    x = np.linspace(0, 2, NX)
    y = np.linspace(0, 2, NY)
    X, Y = np.meshgrid(x, y)

    u = np.zeros((NY, NX))
    v = np.zeros((NY, NX))
    p = np.zeros((NY, NX))

    with open('u.dat', 'r') as f:
        uraw = f.readlines()
    with open('v.dat', 'r') as f:
        vraw = f.readlines()
    with open('p.dat', 'r') as f:
        praw = f.readlines()
    
    fig, ax = plt.subplots()

    def update_quiver(n):
        ax.clear()
        u_flattened = [float(val) for val in uraw[n].strip().split() if val]
        v_flattened = [float(val) for val in vraw[n].strip().split() if val]
        p_flattened = [float(val) for val in praw[n].strip().split() if val]

        for j in range(NY):
            for i in range(NX):
                u[j, i] = u_flattened[j * NX + i]
                v[j, i] = v_flattened[j * NX + i]
                p[j, i] = p_flattened[j * NX + i]

        contour = ax.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
        quiver = ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
        ax.set_title(f'C++, n = {n}')

        return contour.collections + [quiver]

    ani = animation.FuncAnimation(fig, update_quiver, frames=len(uraw), blit=False)
    ani.save('output.gif', writer='imagemagick', fps=5)

if __name__ == '__main__':
    main()
