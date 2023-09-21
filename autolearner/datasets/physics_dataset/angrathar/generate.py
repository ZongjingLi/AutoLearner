import taichi as ti

ti.init()

N = 12
dt = 1e-5

x = ti.Vector.field(2, dtype=ti.f32, shape=(N),needs_grad = True)
v = ti.Vector.field(2, dtype=ti.f32, shape=(N))
U = ti.field(dtype=ti.f32, shape=(), needs_grad = True)

@ti.kernel
def compute_U():
    for i,j in ti.ndrange(N,N):
        x_dir = x[i] - x[j]
        U[None] += -1/x_dir.norm(1e-3)

@ti.kernel
def advance():
    for i in x:
        v[i] += -dt * x.grad[i]
    for i in x:
        x[i] += dt * v[i] 

def substep():
    with ti.ad.Tape(loss=U):
        compute_U()
    advance()

@ti.kernel
def init():
    for i in x: x[i] = [ti.random(), ti.random()]

import numpy as np

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return np.array(list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))

def rgb_to_hex(rgb):
    return '0x%02x%02x%02x' % rgb

def gradient_color(c1,c2,t):
    rgbc1 = hex_to_rgb(c1)/255.
    rgbc2 = hex_to_rgb(c2)/255.
    grad_color = rgbc1*t + rgbc2 * (1-t)
    tc = []
    for i in range(3):
        tc.append(int(grad_color[i] * 255))
    tc = tuple(tc)
    hexc = rgb_to_hex(tc)
    return hexc

import math
def sigma(t):return 1/(1 + math.exp(0-t))

init()

gui = ti.GUI("Visualize", res=(1024,1024))
gui.background_color = 0x0d1926#0x204060
while gui.running:
    for i in range(50):
        substep()
    for i in range(N):
        for j in range(N):
            start = x[i]; end = x[j]
            if i != j:
                s = 7.0
                x_dir = x[i] - x[j]
                t = 1/(1 * x_dir.norm(1e-3))
                t = sigma(s * (t-5))
                gc = gradient_color("#ffffff","#0d1926",t)
                gui.line(start,end,\
                    color= int(gc,0))
    gui.circles(x.to_numpy(), radius = 7,)#color=0x0d1926)
    gui.show()