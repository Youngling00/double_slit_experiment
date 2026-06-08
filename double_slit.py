


import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
import random

wavelength = 0.5
k = 2 * math.pi / wavelength
screen_distance = 6.0
slit_center_y = 0.6
slit_width = 0.12
slit_samples = 60
screen_size = 4.0
screen_points = 500
source_x = -8.0
source_y = 0.0
photons_per_frame = 8
trace_speed = 0.05

slit1_y = -slit_center_y
slit2_y = slit_center_y
slit1_points = np.linspace(slit1_y - slit_width/2, slit1_y + slit_width/2, slit_samples)
slit2_points = np.linspace(slit2_y - slit_width/2, slit2_y + slit_width/2, slit_samples)
screen_y = np.linspace(-screen_size, screen_size, screen_points)
slit_plane_x = 0.0
screen_x = slit_plane_x + screen_distance

def amplitude_from_slit(points, sx, sy, rx):
    a = np.zeros_like(sy, dtype=np.complex128)
    for p in points:
        r = np.sqrt((rx - sx)**2 + (sy - p)**2)
        a += np.exp(1j * k * r) / (r + 1e-9)
    return a

amp1 = amplitude_from_slit(slit1_points, slit_plane_x, screen_y, screen_x)
amp2 = amplitude_from_slit(slit2_points, slit_plane_x, screen_y, screen_x)
amp = amp1 + amp2
intensity = np.abs(amp)**2
trapz = getattr(np, 'trapz', np.trapezoid)
prob = intensity / trapz(intensity, screen_y)

intensity1 = np.abs(amp1)**2
intensity2 = np.abs(amp2)**2
frac1 = intensity1 / (intensity1 + intensity2 + 1e-12)

cdf = np.cumsum(prob)
cdf /= cdf[-1]

fig = plt.figure(figsize=(10, 6))
ax0 = fig.add_axes([0.05, 0.12, 0.45, 0.8])
ax1 = fig.add_axes([0.55, 0.12, 0.40, 0.8])
ax0.set_xlim(source_x - 1.0, screen_x + 1.0)
ax0.set_ylim(-screen_size*1.1, screen_size*1.1)
ax0.set_aspect('equal')
ax0.axis('off')

wall_thickness = 0.5
ax0.add_patch(Rectangle((slit_plane_x - wall_thickness, -screen_size*1.2), wall_thickness, screen_size*2.4, color='saddlebrown'))
ax0.add_patch(Rectangle((slit_plane_x - wall_thickness, slit1_y + slit_width/2), wall_thickness*0.9, screen_size*1.2 - (slit1_y + slit_width/2), color='saddlebrown'))
ax0.add_patch(Rectangle((slit_plane_x - wall_thickness, -screen_size*1.2), wall_thickness*0.9, (slit1_y - slit_width/2) + screen_size*1.2, color='saddlebrown'))
ax0.add_patch(Rectangle((slit_plane_x - wall_thickness, slit2_y + slit_width/2), wall_thickness*0.9, screen_size*1.2 - (slit2_y + slit_width/2), color='saddlebrown'))
ax0.add_patch(Rectangle((slit_plane_x - wall_thickness, -screen_size*1.2), wall_thickness*0.9, (slit2_y - slit_width/2) + screen_size*1.2, color='saddlebrown'))

ax0.plot([source_x], [source_y], marker='o', color='cyan', markersize=8)
ax0.text(source_x - 0.7, source_y + screen_size*1.05, 'Quelle (Photonen)', color='cyan')
ax0.plot([screen_x, screen_x], [-screen_size, screen_size], color='gray', linewidth=2)
ax0.text(screen_x + 0.3, screen_size*0.9, 'Schirm', color='gray')

accum = np.zeros_like(screen_y)
detected = []
moving = []


running = True

line_theo, = ax1.plot(screen_y, intensity / intensity.max(), color='black')
ax1.set_xlabel('Schirm Y')
ax1.set_ylabel('Normierte Intensität / Aufsummierte Treffer')
ax1.set_xlim(-screen_size, screen_size)
ax1.set_ylim(0, 1.6)
line_acc, = ax1.plot([], [], color='blue', linewidth=1.5)
scat = ax0.scatter([], [], s=18, color='blue', alpha=0.8)

def sample_y():
    r = random.random()
    idx = np.searchsorted(cdf, r)
    idx = np.clip(idx, 0, len(screen_y)-1)
    y = screen_y[idx]
    p1 = frac1[idx]
    if random.random() < p1:
        sy = random.uniform(slit1_points[0], slit1_points[-1])
    else:
        sy = random.uniform(slit2_points[0], slit2_points[-1])
    return y, sy

def make_trace(y_scr, sy):
    return {'xs': [source_x, 0.0, screen_x], 'ys': [source_y, sy, y_scr], 't': 0.0}

def update(frame):
    global running
    if running:
        for _ in range(int(max(1, photons_per_frame))):
            y, sy = sample_y()
            moving.append(make_trace(y, sy))

    finished = []
    for i, m in enumerate(moving):
        m['t'] += trace_speed
        if m['t'] >= 1.0:
            ydet = m['ys'][-1]
            idx = np.searchsorted(screen_y, ydet)
            idx = np.clip(idx, 0, len(screen_y)-1)
            accum[idx] += 1
            detected.append(ydet)
            finished.append(i)
    for i in reversed(finished):
        moving.pop(i)

    if detected:
        xs = [screen_x] * len(detected)
        scat.set_offsets(np.column_stack((xs, detected)))
    else:
        scat.set_offsets(np.column_stack(([], [])))

    for line in ax0.lines[:]:
        try:
            if line.get_label() == 'trace':
                line.remove()
        except Exception:
            pass

    for m in moving:
        t = m['t']
        if t < 0.5:
            tt = t / 0.5
            x = (1-tt)*m['xs'][0] + tt*m['xs'][1]
            y = (1-tt)*m['ys'][0] + tt*m['ys'][1]
        else:
            tt = (t-0.5)/0.5
            x = (1-tt)*m['xs'][1] + tt*m['xs'][2]
            y = (1-tt)*m['ys'][1] + tt*m['ys'][2]
        ax0.plot([x], [y], marker='o', color='cyan', markersize=4, label='trace')

    if accum.sum() > 0:
        line_acc.set_data(screen_y, accum / accum.max())
    else:
        line_acc.set_data([], [])

    return scat, line_acc

ani = animation.FuncAnimation(fig, update, frames=2000, interval=40, blit=False)

 
ax_start = fig.add_axes([0.05, 0.01, 0.10, 0.05])
ax_reset = fig.add_axes([0.17, 0.01, 0.10, 0.05])
ax_phot = fig.add_axes([0.55, 0.02, 0.30, 0.03])
ax_speed = fig.add_axes([0.55, 0.06, 0.30, 0.03])

btn_start = Button(ax_start, 'Pause')
btn_reset = Button(ax_reset, 'Zurücksetzen')
slider_phot = Slider(ax_phot, 'Photonen/schritt', 1, 50, valinit=photons_per_frame, valstep=1)
slider_speed = Slider(ax_speed, 'Spurgeschw.', 0.01, 0.2, valinit=trace_speed)

status_text = fig.text(0.5, 0.965, '', ha='center', fontsize=10, color='darkgreen')

def update_status():
    status = 'Läuft' if running else 'Pausiert'
    status_text.set_text(f'Status: {status} — Treffer gesamt: {int(accum.sum())}')

def on_start_clicked(event):
    global running
    running = not running
    btn_start.label.set_text('Pause' if running else 'Start')
    update_status()

def on_reset_clicked(event):
    global accum, detected, moving
    accum = np.zeros_like(screen_y)
    detected = []
    moving = []
    line_acc.set_data([], [])
    scat.set_offsets(np.column_stack(([], [])))
    update_status()

def on_phot_change(val):
    global photons_per_frame
    photons_per_frame = val

def on_speed_change(val):
    global trace_speed
    trace_speed = val

btn_start.on_clicked(on_start_clicked)
btn_reset.on_clicked(on_reset_clicked)
slider_phot.on_changed(on_phot_change)
slider_speed.on_changed(on_speed_change)

update_status()

if __name__ == '__main__':
    plt.suptitle('Doppelspalt-Experiment — Photonen als blaue Punkte')
    plt.show()
