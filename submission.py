import random
import numpy as np

def agent_smit(obs, conf):
    def find_pref(matrix, pref):
        found = False
        rows = matrix.shape[0]

        for r in reversed(range(rows)):
            if matrix[r][pref] == 0:
                found = True
                break
        if found == False:
            return find_pref(matrix, pref + random.choice([-1, 1]))
        return pref
    matrix = np.array(obs.board).reshape(conf.rows, conf.columns)
    centre_col = conf.columns // 2
    return find_pref(matrix, centre_col)
