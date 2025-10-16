# script test raysect

from matplotlib.pyplot import *
from numpy import sqrt, cos
import time
from raysect.optical import World, translate, rotate, rotate_z, Point3D, ConstantSF
from raysect.optical.observer import PinholeCamera, RGBPipeline2D, RGBAdaptiveSampler2D
from raysect.optical.material import VolumeTransform, InhomogeneousVolumeEmitter
from raysect.optical.library import RoughTitanium
from raysect.primitive import Box, Sphere
from raysect.optical.material import Lambert, UniformSurfaceEmitter
from raysect.optical.library.spectra.colours import *
colours = [yellow, orange, red_orange, red, purple, blue, light_blue, cyan, green]


# scene
world = World()
# Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=Lambert(ConstantSF(0.5)))
# boxes

sphere = Sphere(0.02, parent = world, material=UniformSurfaceEmitter(colours[5]))
box_unshifted = Box(
    Point3D(0.2, 0, 0), Point3D(0.4, 1, 1),
    material=UniformSurfaceEmitter(colours[0]),
    parent=world, transform=translate(0, 0, 0) * rotate(0, 0, 0)
)

box_down_shifted = Box(
    Point3D(0, 0, 0), Point3D(0.2, 1, 1),
    material=UniformSurfaceEmitter(colours[1]),
    parent=world, transform=translate(0, 0, 0) * rotate_z(-90)
)

box_up_shifted = Box(
    Point3D(0.2, 0, 0), Point3D(0.4, 1, 1),
    material=UniformSurfaceEmitter(colours[6]),
    parent=world, transform=translate(0, 0, 0) * rotate(0, 0, 90)
)

box_rotated = Box(
    Point3D(-1, -1, -0.25), Point3D(1, 1, 1),
    material=UniformSurfaceEmitter(colours[3]),
    parent=world, transform=translate(-3.2, 1, 0) * rotate(30, 0, 0)
)

# floor = Box(Point3D(-100, -0.1, -100), Point3D(100, 0, 100), world, material=RoughTitanium(0.1))

# camera
rgb = RGBPipeline2D()
sampler = RGBAdaptiveSampler2D(rgb, min_samples=100, fraction=0.2, cutoff=0.01)
camera = PinholeCamera((128, 256), parent=world, transform=translate(0, 0, -10) * rotate(0, 0, 0), pipelines=[rgb], frame_sampler=sampler)
camera.spectral_rays = 1
camera.spectral_bins = 15
camera.pixel_samples = 250

# # integration resolution
# box_unshifted.material.integrator.step = 0.05
# box_down_shifted.material.material.integrator.step = 0.05
# box_up_shifted.material.material.integrator.step = 0.05
# box_rotated.material.material.integrator.step = 0.05

# start ray tracing
ion()
name = 'transform test'
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
render_pass = 1
camera.observe()
rgb.save("{}_{}_pass_{}.png".format(name, timestamp, render_pass))
# while not camera.render_complete:

#     print("Rendering pass {}...".format(render_pass))
#     camera.observe()
#     rgb.save("{}_{}_pass_{}.png".format(name, timestamp, render_pass))
#     print()

#     render_pass += 1

ioff()
rgb.display()