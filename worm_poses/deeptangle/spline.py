#%%

import numpy
import scipy.ndimage
import itertools
import copy

import time # __timing__
#%%
# Convention:
# If point in a 3D array,
# point[0]=x coordinate
# point[1]=y coordinate
# point[2]=z/t coordinate

class B3():
    def __init__(self):
        self.support=4.0

    def value(self, x):
        val = 0.0
        if 0 <= abs(x) and abs(x) < 1:
            val = 2.0 / 3.0 - (abs(x) ** 2) + (abs(x) ** 3) / 2.0
        elif 1 <= abs(x) and abs(x) <= 2:
            val = ((2.0 - abs(x)) ** 3) / 6.0
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if 0 <= x and x < 1:
            val = -2.0 * x + 1.5 * x * x
        elif -1 < x and x < 0:
            val = -2.0 * x - 1.5 * x * x
        elif 1 <= x and x <= 2:
            val = -0.5 * ((2.0 - x) ** 2)
        elif -2 <= x and x <= -1:
            val = 0.5 * ((2.0 + x) ** 2)
        return val

class SplineSheet:
    def __init__(self, M):
        if len(M)!=2:
            raise RuntimeError('M must be a doublet.')

        self.splineGenerator = (B3(), B3())

        if M[0] >= self.splineGenerator[0].support and M[1] >= self.splineGenerator[1].support:
            self.M = M
        else:
            raise RuntimeError('Each M must be greater or equal than its spline generator support size.')

        self.halfSupport = (self.splineGenerator[0].support / 2.0, self.splineGenerator[1].support / 2.0)
        self.controlPoints = numpy.zeros(((self.M[1]+int(self.splineGenerator[1].support))*(self.M[0]+int(self.splineGenerator[0].support)),3))

    def initializeFromPoints(self, samplePoints, sampleParameters):
        # samplePoints is an N x 3 array containing the surface samples to interpolate
        # sampleParameters is an N x 2 array containing the values of s and t corresponding to the samples in samplePoints
        if len(samplePoints)!=len(sampleParameters):
            raise RuntimeError('samplePoints and sampleParameters must be the same length.')

        if len(samplePoints[0])!=3:
            raise RuntimeError('samplePoints must contain triplets.')

        if len(sampleParameters[0])!=2:
            raise RuntimeError('sampleParameters must contain doublets.')

        N=len(samplePoints)
        phi=numpy.zeros((N,(self.M[1]+int(self.splineGenerator[1].support))*(self.M[0]+int(self.splineGenerator[0].support))))
        sigmaX=numpy.zeros((N))
        sigmaY=numpy.zeros((N))
        sigmaZ=numpy.zeros((N))

        for n in range(0, N):
            sigmaX[n] = samplePoints[n,0]
            sigmaY[n] = samplePoints[n,1]
            sigmaZ[n] = samplePoints[n,2]

            t = sampleParameters[n,1]
            s = sampleParameters[n,0]

            for l in range(0,self.M[0]+int(self.splineGenerator[0].support)):
                for k in range(0,self.M[1]+int(self.splineGenerator[1].support)):
                    kp = k + ((self.M[1]+int(self.splineGenerator[1].support)) * l)

                    tVal = t - (k - self.halfSupport[1])
                    if (tVal > -self.halfSupport[1]) and (tVal < self.halfSupport[1]):
                        basisFactor1 = self.splineGenerator[1].value(tVal)
                    else:
                        basisFactor1 = 0

                    sVal = s - (l - self.halfSupport[0])
                    if (sVal > -self.halfSupport[0]) and (sVal < self.halfSupport[0]):
                        basisFactor2 = self.splineGenerator[0].value(sVal)
                    else:
                        basisFactor2 = 0

                    phi[n,kp] = phi[n,kp] + (basisFactor1 * basisFactor2)

        cX=numpy.linalg.lstsq(phi,sigmaX, rcond = None)
        cY=numpy.linalg.lstsq(phi,sigmaY, rcond = None)
        cZ=numpy.linalg.lstsq(phi,sigmaZ, rcond = None)

        for l in range(0,self.M[0]+int(self.splineGenerator[0].support)):
            for k in range(0,self.M[1]+int(self.splineGenerator[1].support)):
                kp = k + ((self.M[1]+int(self.splineGenerator[1].support)) * l)
                self.controlPoints[kp] = numpy.array([cX[0][kp], cY[0][kp], cZ[0][kp]])

        return

    def centroid(self):
        centroid=numpy.zeros((3))

        for k in range(0,len(self.controlPoints)):
            centroid+=self.controlPoints[k]

        return centroid/len(self.controlPoints)

    def scale(self, scalingFactor):
        centroid=self.centroid()

        for k in range(0,len(self.controlPoints)):
            vectorToCentroid=self.controlPoints[k]-centroid
            self.controlPoints[k]=centroid+scalingFactor*vectorToCentroid

        return

    def sample(self, samplingRate):
        if len(samplingRate)!=2:
            raise RuntimeError('samplingRate must be a doublet.')

        samplingT = range((self.M[1] - 1) * samplingRate[1] + 1)
        samplingS = range((self.M[0] - 1) * samplingRate[0] + 1)
        samplinglist = list(itertools.product(samplingS, samplingT))
        paramlist = [(float(x[0]) / samplingRate[0], float(x[1]) / samplingRate[1]) for x in samplinglist]

        surfacePoints = [self.parametersToWorld(p) for p in paramlist]
        surfacePoints = numpy.asarray(surfacePoints)

        return (paramlist, surfacePoints)

    def parametersToWorld(self, params, d=(False,False)):
        point = numpy.zeros((3))
        point = self.nonPoleContributions(params[1], params[0], point, d[1], d[0])

        if numpy.any(numpy.isnan(point)):
            error('ERROR: parametersToWorld returned NaN value.')

        return point

    def nonPoleContributions(self, t, s, point, dt=False, ds=False):
        for l in range(0, self.M[0]+int(self.splineGenerator[0].support)):
            sVal = s - (l - self.halfSupport[0])
            if (sVal > -self.halfSupport[0]) and (sVal < self.halfSupport[0]):
                for k in range(0, self.M[1]+int(self.splineGenerator[1].support)):
                    tVal = t - (k - self.halfSupport[1])
                    if (tVal > -self.halfSupport[1]) and (tVal < self.halfSupport[1]):
                        if ds and dt:
                            point=point+(self.controlPoints[k + (l * (self.M[1]+int(self.splineGenerator[1].support)))]*self.splineGenerator[0].firstDerivativeValue(sVal)*self.splineGenerator[1].firstDerivativeValue(tVal))
                        elif ds:
                            point=point+(self.controlPoints[k + (l * (self.M[1]+int(self.splineGenerator[1].support)))]*self.splineGenerator[0].firstDerivativeValue(sVal)*self.splineGenerator[1].value(tVal))
                        elif dt:
                            point=point+(self.controlPoints[k + (l * (self.M[1]+int(self.splineGenerator[1].support)))]*self.splineGenerator[0].value(sVal)*self.splineGenerator[1].firstDerivativeValue(tVal))
                        else:
                            point=point+(self.controlPoints[k + (l * (self.M[1]+int(self.splineGenerator[1].support)))]*self.splineGenerator[0].value(sVal)*self.splineGenerator[1].value(tVal))
        return point

