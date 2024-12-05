from typing import List, Tuple, Dict, Any, Union, Optional, Iterable
import numpy as np
import math, random

def fourierFeatureExtractor2(
        state,
        maxCoeff: int = 5,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), maxCoeff 2, and scale [2, 1, 1], this should output (in any order):
    [1, cos(pi * 2x), cos(pi * y), cos(pi * z),
     cos(pi * (2x + y)), cos(pi * (2x + z)), cos(pi * (y + z)),
     cos(pi * (4x)), cos(pi * (2y)), cos(pi * (2z)),
     cos(pi*(4x + y)), cos(pi * (4x + z)), ..., cos(pi * (4x + 2y + 2z))]
    '''
    if scale is None:
        scale = np.ones_like(state)
    features = None

    allTerms = np.arange(1)
    stateScaled = [v * scale[idx] for idx, v in enumerate(state)]
    for s in stateScaled:
        sTerms = s * np.arange(maxCoeff + 1)
        newAllTerms = np.arange(0)
        for sTerm in sTerms:
            newAllTerms = np.append(newAllTerms, sTerm + allTerms)
        allTerms = newAllTerms
    for idx, value in np.ndenumerate(allTerms):
        allTerms[idx] = math.cos(math.pi * value)
    features = allTerms

    return features

def fourierFeatureExtractor(
        state,
        maxCoeff: int = 5,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), maxCoeff 2, and scale [2, 1, 1], this should output (in any order):
    [1, cos(pi * 2x), cos(pi * y), cos(pi * z),
     cos(pi * (2x + y)), cos(pi * (2x + z)), cos(pi * (y + z)),
     cos(pi * (4x)), cos(pi * (2y)), cos(pi * (2z)),
     cos(pi*(4x + y)), cos(pi * (4x + z)), ..., cos(pi * (4x + 2y + 2z))]
    '''
    if scale is None:
        scale = np.ones_like(state)
    features = None

    allTerms = np.arange(1)
    stateScaled = [v * scale[idx] for idx, v in enumerate(state)]
    for s in stateScaled:
        sTerms = s * np.arange(maxCoeff + 1)
        allTerms = allTerms.reshape(len(allTerms), 1) + sTerms.reshape(1, len(sTerms))
        allTerms = allTerms.flatten()

    for idx, value in np.ndenumerate(allTerms):
        allTerms[idx] = math.cos(math.pi * value)

    return allTerms

def polynomialFeatureExtractor(
        state,
        degree: int = 3,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), degree 2, and scale [2, 1, 1], this should output:
    [1, 2x, y, z, 4x^2, y^2, z^2, 2xy, 2xz, yz, 4x^2y, 4x^2z, ..., 4x^2y^2z^2]
    '''
    if scale is None:
        scale = np.ones_like(state)

    # Create [1, s[0], s[0]^2, ..., s[0]^(degree)] array of shape (degree+1,)
    firstPolyFeat = (state[0] * scale[0])**(np.arange(degree + 1))
    currPolyFeat = firstPolyFeat

    for i in range(1, len(state)):
        # Create [1, s[i], s[i]^2, ..., s[i]^(degree)] array of shape (degree+1,)
        newPolyFeat = (state[i] * scale[i])**(np.arange(degree + 1))

        # Do shape (len(currPolyFeat), 1) times shape (1, degree+1) multiplication
        # to get broadcasted result of shape (len(currPolyFeat), degree+1)
        # Note that this is also known as the vector outer product.
        currPolyFeat = currPolyFeat.reshape((len(currPolyFeat), 1)) * newPolyFeat.reshape((1, degree + 1))

        # Flatten to (len(currPolyFeat) * (degree+1),) array for the next iteration or final features.
        currPolyFeat = currPolyFeat.flatten()
    return currPolyFeat

if __name__ == "__main__":
    state = [10, 20, 20, 1, 3, 3000]
    a = fourierFeatureExtractor(state, 2)
    b = fourierFeatureExtractor2(state, 2)
    print(a)
    a.sort()
    b.sort()
    print(f"a == b? {a == b}")