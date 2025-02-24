# Inverse Kinematics on Guiding Vector Fields for Robot Path Following

## Research paper

**ABSTRACT:** Inverse kinematics is a fundamental technique for
motion and positioning control in robotics, typically applied
to end-effectors. In this paper, we extend the concept of
inverse kinematics to guiding vector fields for path following
in autonomous mobile robots. The desired path is defined by
its implicit equation, i.e., by a collection of points belonging to
one or more level sets. These level sets serve as a reference to
construct an error signal that drives the guiding vector field
toward the desired path, enabling the robot to converge and
travel along the path by following such vector field. We leverage
inverse kinematics to ensure that the constructed error signal
behaves like a linear system, facilitating control over the robot’s
transient motion toward the desired path and allowing for
the injection of feed-forward signals to induce precise motion-
behavior along the path. We start with the formal exposition
on how inverse kinematics can be applied to single-integrator
robots in an m-dimensional Euclidean space. We then propose
solutions to the theoretical and practical challenges of applying
this technique to unicycles with constant speeds to follow 2D
paths with precise transient control. We finish by validating the
formal results with experiments using fixed-wing drones.

    @misc{yuzhoujesusbautista2024ikgvf,
      title={Inverse Kinematics on Guiding Vector Fields for Robot Path Following}, 
      author={Yu Zhou, Jesus Bautista, Weijia Yao, Hector Garcia de Marina},
      year={2024},
      url={}, 
    }

This paper is in the proceeding of the IEEE International Conference on Robotics and Automation (ICRA) 2025.

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