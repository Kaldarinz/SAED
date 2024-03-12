import numpy as np

def calc_spectrum(
        data: np.ndarray,
        coords: np.ndarray,
        scale: float,
        dr: int = 5,
        )-> np.ndarray:
    """
    Calculate SAED spectrum.
    
    PARAMETERS
    ----------
    `data` - array with SAED spectrum.\n
    `coords` - array with distances from center of SAED data.\n
    `dr` - step of averaging.\n
    `scale` - conversion coeff from pizels to 1/nm.

    RETURN
    ------
    2d array, where first column is coordinates in 1/nm, while second 
    column is average intensity if the current step.
    """

    start = 20

    # Output array
    result = np.zeros((int(coords.max()) - start, 2))
    for i in range(result.shape[0]):
        r = start + i
        min_r = r - int(dr/2)
        max_r = min_r + dr
        cur = np.zeros(coords.shape)
        cur[(coords<max_r)*(coords>min_r)] = 1
        # Average intensity at current distance
        result[i,1] = data[cur>0].sum()/cur.sum()
        result[i,0] = r/scale
    print('Spectrum calculated')
    return result

def optimize_position(
        data: np.ndarray,
        start: tuple,
        x_range: int,
        y_range: int,
        dr: int = 4,
        scale: float = 40.4):
    """
    Find optimal center position of SAED data.
    
    PARAMETERS
    ----------
    `data` - array with SAED spectrum.\n
    `start` - initial coordinate (x,y) for search.\n
    `x_range` - range for optimization of X coordinate.\n
    `y_range` - range for optimization of Y coordinate.\n
    `dr` - step for SAED spectrum calculation.\n
    `scale` - conversion coeff from pizels to 1/nm.
    """
     
    # Max value in a spectrum
    max_val = -1
    # Optimal spectrum
    spectrum = []
    # Optimal center of SAED data
    new_cent = start
    for i in range(start[0] - int(x_range/2), start[0] + int(x_range/2) + 1):
        for j in range(start[1] - int(y_range/2), start[1] + int(y_range/2) + 1):
            print(f'Start SAED calculation for (x,y) = {(i,j)}')
            x = np.arange(-i, data.shape[0] - i)
            y = np.arange(-j, data.shape[1] - j)
            xs, ys = np.meshgrid(x, y, sparse=True, indexing = 'xy')
            zs = np.sqrt(xs**2 + ys**2)
            print(zs.shape)
            cur_res = calc_spectrum(data = data, coords = zs, dr = dr, scale = scale)
            if (cur_mv:=cur_res.max()) > max_val:
                print('New max val = ', cur_mv)
                print('New center (x,y) = ', (i, j))
                max_val = cur_mv
                spectrum = cur_res.copy()
                new_cent = (i, j)
    
    return spectrum, new_cent

if __name__ == '__main__':
    pass