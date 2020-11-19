# This program takes a matrix and filter dimension 
# and performs mean filtering in spatial domain
# saving the results.
import os
import numpy as np

# Spatial domain filtering
# Getting the matrix dimension to apply the filter
print('The matrix will be generated as:')
print(np.arange(1, 10).reshape(3, 3))
print('Enter the matrix dimension:')

# Making sure the user entered a valid integer
while True:
    try:
        n = int(input())
        break
    except:
        print('Enter a valid integer.\n')
        
# Generating the matrix and printing it
userMatrix = np.arange(1, n*n + 1).reshape(n, n)
print('\nYour matrix:')
print(userMatrix)

# Getting filter dimension
print('\nEnter the mean filter dimension:')

# Making sure the user entered a valid odd integer 
while True:
    try:
        m = int(input())
        if m < n and ((m % 2) == 1):
            break
        elif m >= n:
            print('m must be less than n (%s).' % n)
        elif not(m % 2):
            print('m needs to be odd.')
    except:
        print('Enter a valid integer.')

# Generating the filter
filt = np.ones((m, m)) / float(m * m)

# Applying zero padding in the matrix
padDim = int((m - 1) / 2)
padMatrix = np.pad(userMatrix, ((padDim, padDim)), 'constant', constant_values=0)
print('\nMatrix after zero padding:\n')
print(padMatrix)

# Generating the filtered matrix
filMatrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sliceMatrix = padMatrix[i : m+i, j: m+j]
        result = np.multiply(filt, sliceMatrix).sum()
        filMatrix[i][j] = result
filMatrix = np.around(filMatrix, 2)
print('\n\nMatrix after filtering:\n')
print('\n', filMatrix)

# Creating folder for saving the results.
os.makedirs('sptDomFilt', exist_ok=True)

cwdPath = os.getcwd()
domPath = os.path.join(cwdPath, 'sptDomFilt')
os.chdir(domPath)

# Saving files.
print('\nSaving files...')
np.save('matrix', userMatrix)
np.save('filter', filt)
np.save('filteredMatrix', filMatrix)
print('\nDone.')

