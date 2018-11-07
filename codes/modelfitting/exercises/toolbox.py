#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
import numpy as np
def perp( a ) :
    # Source: https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2 [y,x]
# line segment b given by endpoints b1, b2 [y,x]
# return
def seg_intersect(a1,a2, b1,b2) :
    a1, a2, b1, b2 = np.asarray(a1), np.asarray(a2), np.asarray(b1), np.asarray(b2)
    # Adapted from source: https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    if denom == 0.:
        return None
    else:
        return (num / denom.astype(float))*db + b1


def line_pts(rho, theta, scale=10000):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + scale * (-b))
    y1 = int(y0 + scale * (a))
    x2 = int(x0 - scale * (-b))
    y2 = int(y0 - scale * (a))

    return x1, y1, x2, y2
