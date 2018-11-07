import numpy as np

particles = None
weights = None
particlesN = 0
newParticleFun = lambda x: x
def init(_newParticleFun, N=100):
    global particlesN, particles, weights
    particlesN = N

    particles = _newParticleFun(particlesN)
    weights = np.zeros(particlesN)

def motionUpdate(_motionFun):
    if particles is not None:
        particles[:] = _motionFun(particles)

def weighting(z, _scoringFun):
    weights[:] = _scoringFun(z, particles)
    weights[:] = weights[:]/np.sum(weights[:])

def resampling(_resamplingFun, _resampledParticles=1.0):
    resampleN = int(particlesN*_resampledParticles)
    newN = particlesN - resampleN

    if particles is not None:
        particles[:resampleN] = _resamplingFun(particles, weights, resampleN)
    if newN > 0:
        particles[resampleN:] = newParticleFun(newN)
    weights[:] = 0.0  # Reset the weights