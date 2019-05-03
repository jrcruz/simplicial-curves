from sys import argv
from numpy import loadtxt
from numpy.linalg import norm
import pylab as PL

sampled_curve = loadtxt(argv[1])

PL.plot(sampled_curve)
PL.savefig(argv[1] + ".derivative.png", bbox_inches="tight")
