# From tutorial at: https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
import cython
cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def some_filter(np.ndarray[DTYPE_t, ndim=2] source):
    # TODO: work out a better way to do this...
    assert source.dtype == DTYPE
    cdef int R = source.shape[0]
    cdef int C = source.shape[1]
    #cdef np.ndarray[DTYPE_t, ndim=2] h = np.zeros([R, C], dtype=DTYPE)

    cdef int r, c, r_min, r_max, c_min, c_max, rr, cc
    cdef int count, count_cardinal, count_diagonal, count_ns, count_we, count_clear_diag

    for r in range(R):
        r_min = max(0, r - 1)
        r_max = min(R, r + 2)
        for c in range(C):
            c_min = max(0, c - 1)
            c_max = min(C, c + 2)
            count = 0
            count_cardinal = 0
            count_diagonal = 0
            count_ns = 0
            count_we = 0
            count_clear_diag = 0
            for rr in range(r_min, r_max):
                for cc in range(c_min, c_max):
                    if source[rr, cc] > 0:
                        count += 1
                        if (rr - r) == 0 or (cc - c) == 0:
                            count_cardinal += 1
                        else:
                            count_diagonal += 1
                        if (rr - r) == 0:
                            count_we += 1
                        if (cc - c) == 0:
                            count_ns += 1
                        if (rr - r) != 0 and (cc - c) != 0: # On a diagonal
                            # Check if the diagonal depends on it
                            if source[r, cc] == 0 and source[rr, c] == 0:
                                count_clear_diag += 1

            
            # if source[r, c] > 0 and (count > 3 or (count_we == 2 and count_ns == 2)):
            if source[r, c] > 0 and (
                    (count_we == 2 and count_ns == 2) # L shape
                    or (count_we == 3 and count_ns == 2) # T shape
                    or (count_we == 2 and count_ns == 3) # T shape
                ) and (
                    count_clear_diag == 0 # Would break connection to diagonal
                ):
                #h[r, c] = 0
                source[r, c] = 0
                #print(str(r) + "," + str(c) + ": " + str(count))
    #return h
    return source
