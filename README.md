# Distributed Oscillatory Guidance for Formation Flight Fixed-Wing Drones

## Research paper

**ABSTRACT:** The autonomous formation flight of fixed-wing drones is hard when the coordination requires actuation over their speeds since they are critically bounded and aircraft are mostly designed to fly at a nominal airspeed. This paper proposes an algorithm to achieve formation flights of fixed-wing drones without requiring any actuation over their speed. In particular, we guide all the drones to travel over specific paths, e.g., parallel straight lines, and we superpose an oscillatory behavior onto the guiding vector field that drives the drones to the paths. This oscillation enables control over the average velocity along the path, thereby facilitating inter-drone coordination. Each drone adjusts its oscillation amplitude distributively in a closed-loop manner by communicating with neighboring agents in an undirected and connected graph. A novel consensus algorithm is introduced, leveraging a non-negative, asymmetric saturation function. This unconventional saturation is justified since \emph{negative} amplitudes do not make drones travel backward or have a negative velocity along the path. Rigorous theoretical analysis of the algorithm is complemented by validation through numerical simulations and a real-world formation flight.

    @misc{xuyangbautistajesus2025ikgvf,
      title={Distributed Oscillatory Guidance for Formation Flight Fixed-Wing Drones}, 
      author={Yang Xu, Jesús Bautista, José Hinojosa, Héctor García de Marina},
      year={2025},
    }
    
## Features
This project includes:

* Simulations of the Inverse Kinematics GVF algorithm
* Visualization of Paparazzi UAV telemetry data from experiments
* Animations of both telemetry and simulation data

## Quick Install

To install, simply run:

```bash
python install.py
```

## Usage

Run the Jupyter notebooks inside the `notebooks` directory.

## Credits

If you have any questions, open an issue or reach out to the maintainers:

- **[Jesús Bautista Villar](https://sites.google.com/view/jbautista-research)** (<jesbauti20@gmail.com>) – Main Developer
