import numpy as np 
import matplotlib.pyplot as plt
from scenarios.orbital_elements.LEO_circ_i15 import scene
import visualization.animation as animation
from visualization.plotter import plot_attitude, plot_ground_track, plot_trajectory
from copy import copy
from sim.bodies import Earth
import math
from tqdm import trange
import datetime

animate = False
plot_pos = False
plot_att = False
plot_ground = True
ground_body = Earth

steps = math.ceil(scene.duration / scene.dt)

def main():
    print(f"Simulating {scene.name} for {datetime.timedelta(seconds=scene.duration)}")

    X = scene.X0
    u = {}
    t = scene.t0

    save_history = plot_pos or plot_att or plot_ground
    if save_history:
        Xhist = [copy(X)]
        Thist = [t]
    if animate:
        animation_plotter = animation.init(f"animations/{scene.name}.mp4", framerate=60)

    try:
        if animate:
            animation.save_frame(animation_plotter, X, scene.cam_target, u=u)

        for k in trange(0, steps):
            if k % scene.dt_between_gnc_updates == 0:
                u = scene.update_gnc(X, t)
            t += scene.dt
            scene.step(X)

            if save_history:
                Xhist.append(copy(X))
                Thist.append(t)
            if animate and k % scene.dt_between_frames == 0:
                animation.save_frame(animation_plotter, X, scene.cam_target, u=u)
    finally:
        if animate:
            animation.close(animation_plotter)
        if plot_pos:
            plot_trajectory(Xhist, f"plots/position/{scene.name}.png", show=False)
        if plot_att:
            plot_attitude(Xhist, f"plots/attitude/{scene.name}.png", show=False)
        if plot_ground:
            plot_ground_track(Xhist, Thist, ground_body,
                              f"plots/ground_track/{scene.name}_{ground_body.name}.png",
                              show=False, rotation_epoch=scene.t0)
        if save_history:
            plt.show()


if __name__ == "__main__":
    main()
