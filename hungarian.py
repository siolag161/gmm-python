import numpy as np
import sys


class Hungarian:
    """ the solver for assignment problem using hungarian method """
    def __init__(self):
        self.rowCover = []
        self.colCover = []
        self.dim = 0
        self.matrix = None
        self.marked = None
        self.path = None
        self.Z0_r = 0
        self.Z0_c = 0
        
    #@staticmethod
    def _pad_to_square(self, matrix, pad_value = 0):
        m = matrix.reshape((matrix.shape[0], -1))
        padded = pad_value * np.ones(2 * [max(m.shape)], dtype=m.dtype)
        padded[0:m.shape[0], 0:m.shape[1]] = m
        return padded

    def _make_matrix(self, matrix):
        m = np.array(matrix)
        self.original = m
        self.original_shape = m.shape
        if m.shape[0] != m.shape[1]:
            m = self._pad_to_square(m)        
        return m

    def _switch_step(self, step):
        self.steps = {1: self._step_1, 2: self._step_2, 3: self._step_3, 
                 4: self._step_4, 5:self. _step_5, 6:self. _step_6, 7: self._step_7}    
        return self.steps[step]

    def _init_matrix(self, matrix):
        self.matrix = self._make_matrix(matrix)
        self.dim = self.matrix.shape[0]
        self.rowCover = 0*np.ones(self.dim, int)
        self.colCover = 0*np.ones(self.dim, int)
        self.marked = 0*np.ones(2*[self.dim], int)
        self.path = 0*np.ones(2*[self.dim*2], int)
        self.changed_sign = False
        
    def compute(self, matrix, maximum=False):        
            
        self._init_matrix(matrix)
        #print self.original
        if (maximum):
            self.matrix = np.max(self.matrix)-self.matrix
            #print self.matrix
        done = False
        step = 1

        while not done:
            done, step = self._switch_step(step)()
            #print done, step, self.marked
            
        rs = self._compute_cost()
        
        return rs

    def _compute_cost(self):
        #print self.original
        results = []
        cost = 0.0
        #print self.marked
        for i in range(self.original_shape[0]):
            for j in range(self.original_shape[1]):
                if self.marked[i,j] == 1:
                    #print '%d->%d' %(i, j)
                    results += [(i, j)]
                    cost += self.original[i][j]
        #print self.matrix
        #print 'cost = %f' %cost
        return results

    def _step_1(self):  
        for i in range(self.dim):
            self.matrix[i] = self.matrix[i] - min(self.matrix[i])
    
        return False, 2
    
    def _step_2(self):
        for r in range(self.dim):
            for c in range(self.matrix.shape[1]):
                if self.matrix[r,c] == 0 and self.rowCover[r] == 0 and self.colCover[c] == 0:
                    self.marked[r,c] = 1
                    self.rowCover[r] = 1
                    self.colCover[c] = 1
                    
        self._clear_covers()
           
        return False, 3
        
    def _step_3(self):
        """ count the number of covered columns"""
        
        for i in range(self.dim):
            for j in range(self.dim):
                if self.marked[i,j] == 1:
                    self.colCover[j] = 1
                    
        count = np.count_nonzero(self.colCover)   

        if count >= self.dim:
            return False, 7
        
        return False, 4

    def _non_covered_zero(self):     
        for i in range(self.dim):
            for j in range(self.dim):
                if self.matrix[i,j] == 0 and self.rowCover[i] == 0 and self.colCover[j] == 0:
                    return (i, j)

        return (-1,-1)
        
    def _find_star_in_row(self, row):
        col = -1
        cols = np.nonzero(self.marked[row, :] == 1)[0]
        if (cols.size > 0):
            col = cols[0]
        return col
       
    def _step_4(self):
        done = False
        row, col = (-1, -1)
        while not done:
            row, col = self._non_covered_zero()
            if (row == -1):
                return False, 6
            else:
                self.marked[row, col] = 2
                c = self._find_star_in_row(row)
                if (c == -1):
                    self.Z0_r, self.Z0_c = (row, col)
                    return False, 5
                else:
                    col = c
                    self.rowCover[row] = 1
                    self.colCover[col] = 0    
                    
    ########################    
    def _find_star_in_col(self, col): 
        row = -1
        rows = np.nonzero(self.marked[:, col] == 1)[0]
        if (rows.size > 0):
            row = rows[0]
        return row
        
    def _find_prime_in_row(self, row):
        col = -1
        for j in range(self.dim):
            if self.marked[row, j] == 2:
                col = j
                break

        return col
                
    def _step_5(self):
        count = 1
        path = self.path
        path[count-1][0] = self.Z0_r
        path[count-1][1] = self.Z0_c
        done = False
        
        while not done:
            row = self._find_star_in_col(path[count-1, 1])
            if row >= 0:
                count += 1
                path[count-1][0] = row
                path[count-1][1] = path[count-2][1]
            else:
                done = True
            if not done:                
                col = self._find_prime_in_row(path[count-1, 0])
                count += 1
                path[count-1][0] = path[count-2, 0]
                path[count-1][1] = col


        self._convert_path(path, count)
        self._clear_covers()
        self._erase_primes()

        return False, 3        

    def _convert_path(self, path, count):
        for p in range(count):
            self.marked[path[p, 0], path[p,1]] = self.marked[path[p, 0], path[p,1]]-1
                
    def _clear_covers(self):
        self.rowCover.fill(0)
        self.colCover.fill(0)

    def _erase_primes(self):
        """Erase all prime markings"""
        for i in range(self.dim):
            for j in range(self.dim):
                if self.marked[i, j] == 2:
                    self.marked[i, j] = 0
                            
    def _step_6(self):
        minval = self._find_smallest()
        
        for i in range(self.dim):
            for j in range(self.dim):
                if self.rowCover[i]:
                    self.matrix[i, j] += minval
                if not self.colCover[j]:
                    self.matrix[i, j] -= minval

        return False, 4
        
    def _find_smallest(self):
        min_val = sys.maxint
        for i in range(self.dim):
            for j in range(self.dim):
                if self.rowCover[i] ==0  and self.colCover[j] == 0:
                    min_val = min(min_val, self.matrix[i,j])
        return min_val                
        
    def _step_7(self):
        return True, 0

