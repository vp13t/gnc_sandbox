import numpy as np 
from scenarios.LEO_circ import scene
import visualization.animation as animation
from visualization.plotter import plot_trajectory
from copy import copy
from sim.bodies import Earth
import math
from tqdm import trange

animate = False
plot = True

dt = 0.1  # Time step in seconds
dt_between_gnc_updates = 1
dt_between_frames = 10
steps = math.ceil(scene.duration / dt)

def main():
    print(f"Simulating {scene.name} for {scene.duration} seconds...")

    X = scene.X0
    u = {}
    t = scene.t0

    if plot:
        Xhist = [copy(X)]
    if animate:
        animation_plotter = animation.init(f"animations/{scene.name}.mp4", framerate=60)

    try:
        if animate:
            animation.save_frame(animation_plotter, X, scene.cam_target, u=u)

        for k in trange(0, steps):
            if k % dt_between_gnc_updates == 0:
                u = scene.update_gnc(X, t)
            t += dt
            X.update(dt, scene.spacecraft.mass, scene.forces(X))

            if plot:
                Xhist.append(copy(X))
            if animate and k % dt_between_frames == 0:
                animation.save_frame(animation_plotter, X, scene.cam_target, u=u)
    finally:
        if animate:
            animation.close(animation_plotter)

    if plot:
        plot_trajectory(Xhist, f"plots/{scene.name}.png", show=True)


if __name__ == "__main__":
    main()
