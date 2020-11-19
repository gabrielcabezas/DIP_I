# Frequency domain filtering.
# Discrete Fourier Transform (DFT) implementation 2-D and 
# separability of the 2-D DFT.

import numpy as np
import cmath
import matplotlib.pyplot as plt
import time


# DFT 2-D
def dft2D (f):

    # Getting the original image dimensions
    try:
        M, N = f.shape
    except:
        print('Image is not bidimensional')
        return
    
    # Generating the Fourier Transform
    F = np.zeros((2 * M, 2 * N), dtype=complex)
    
    # Applying the algorithm:
    for u in range(2 * M):
        for v in range(2 * N):
            for x in range(M):
                for y in range(N):
                    
                    # Polar form r = f[x][y], phi = -2pi((u*x)/M + (v*y)/N))
                    res = cmath.rect(f[x][y], (-2)*np.pi*((u*x)/M + (v*y)/N))
                    F[u][v] += res

    
    # Rounding F value
    F = np.around(F, 2)

    # Getting the slice values for centering F
    if (M%2):
        widL = (M-1) // 2
        widR = (M-1) // 2
    else:
        widL = M // 2
        widR = widL - 1

    if (N%2):
        heiU = (N-1) // 2
        heiD = (N-1) // 2
    else:
        heiU = N // 2
        heiD = heiU - 1
 
    sliceF = F.copy()
    sliceF = sliceF[M - widL : M + widR + 1 , N - heiU : N + heiD + 1]
    
    # Return the centered transform
    return sliceF

# DFT 1-D
def dft1D (f):
    
    # Getting vector dimension
    try:
        M = f.shape[0]
    except:
        print('Vector is not unidimensional')
        return
    
    # Generating Fourier transform
    F = np.zeros(M, dtype=complex)

    for m in range(M):
        for n in range(M):
            res = f[n] * cmath.rect(1 , (-2)*np.pi*(m*n)/M)            
            F[m] += res

    # Rounding F value
    F = np.around(F, 2)
    
    # Return the transform
    return  F

# Applying the separability to calculate 2-D DFT
def dft2DSep (f):
    
    # Getting original image dimensions
    try:
        M, N = f.shape
    except:
        print('Image is not bidimensional')
        return
    
    # Generating Fourier transform
    F_rows = np.zeros((2 * M, 2 * N), dtype=complex)

    # Applying 1-D DFT on rows
    for i in range(M):
        F_rows[i] += np.concatenate((dft1D(f[i]), dft1D(f[i])))
        F_rows[i + M] += np.concatenate((dft1D(f[i]), dft1D(f[i])))
    
    # Transposing the matrix so that columns become rows
    F_rows = F_rows.T
    
    F = np.zeros((2 * M, 2 * N), dtype=complex)

    # Applying 1-D DFT on columns
    for i in range(N):
        F[i] += np.concatenate((dft1D(F_rows[i][:N]), dft1D(F_rows[i][:N])))
        F[i + N] += np.concatenate((dft1D(F_rows[i][:N]), dft1D(F_rows[i][:N])))

    # Transposing back the matrix
    F = F.T
    
    # Getting slice values for centering F
    if (M%2):
        widL = (M-1) // 2
        widR = (M-1) // 2
    else:
        widL = M // 2
        widR = widL - 1

    if (N%2):
        heiU = (N-1) // 2
        heiD = (N-1) // 2
    else:
        heiU = N // 2
        heiD = heiU - 1
 
    sliceF = F.copy()
    sliceF = sliceF[M - widL : M + widR + 1 , N - heiU : N + heiD + 1]
    sliceF = np.around(sliceF, 2)

    # Return the transform
    return sliceF

# Plotting DFT Spectrum
def plotDFTSpectrum(F, c, name=None):
    spectrum = c * np.log(np.abs(F))
    plt.imshow(spectrum, cmap='gray')
    if name:
        plt.title(name)

# Computing DFT of randomly generated matrix
np.random.seed(42)
for sizeOfA in [(5, 5), (20, 20)]:

    # Generating the input
    A = np.random.randint(0, 256, size=sizeOfA)

    # Computing runtime of each algorithm
    start = time.time()
    F_dft = dft2D(A)
    DFTtime = np.around(time.time() - start, 4)

    start = time.time()
    F_sep = dft2DSep(A)
    SepTime = np.around(time.time() - start, 4)

    # Applying the FFT algorithm
    start = time.time()
    F_fft = np.around(np.fft.fftshift(np.fft.fft2(A)), 2)
    FFTtime = np.around(time.time() - start, 4)

    # Printing runtime
    print('For dimension %s x %s:\n' % sizeOfA)
    print('DFT 2D time: %s sec.\n' % DFTtime)
    print('DFT2D Sep time: %s sec.\n' % SepTime)
    print('FFT time: %s sec.\n' % FFTtime)

    # Plotting the transform spectrum
    plt.subplot(1, 3, 1)
    plotDFTSpectrum(F_fft, 20, 'FFT Spectrum')
    plt.subplot(1, 3, 2)
    plotDFTSpectrum(F_dft, 20, 'DFT2D Spectrum')
    plt.subplot(1, 3, 3)
    plotDFTSpectrum(F_sep, 20, 'DFT2D Sep Spectrum')
    plt.show()

    # Sum of squared errors comparing FFT and the other algorithms
    resDFT2D = np.square(np.abs(F_dft - F_fft)).sum()
    resDFTSep = np.square(np.abs(F_sep - F_fft)).sum()
    print('SSE for DFT2D: %s\n' % resDFT2D)
    print('SSE for DFT2DSep: %s\n\n' % resDFTSep)
    print('*' * 20, '\n')


