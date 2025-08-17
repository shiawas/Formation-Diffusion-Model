import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_rectangle(param, style, hdl=None):
    """
    Draw a rotated rectangle using matplotlib.

    param: [a, b, w, h, theta]
        - (a, b): center of rectangle
        - (w, h): width and height
        - theta: rotation angle (radians)
    style: dict
        - 'facecolor': fill color
        - 'edgecolor': border color
        - 'linewidth': border width
    hdl: matplotlib Polygon object (optional), to update instead of redraw
    """

    a, b, w, h, theta = param

    # rectangle centered at (0,0)
    X = np.array([-w/2, w/2, w/2, -w/2, -w/2])
    Y = np.array([ h/2, h/2, -h/2, -h/2,  h/2])
    P = np.vstack([X, Y])

    # rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    P_rot = R @ P

    # translate
    X_rot = P_rot[0, :] + a
    Y_rot = P_rot[1, :] + b

    if hdl is None:
        # draw polygon
        hdl, = plt.fill(X_rot, Y_rot, facecolor=style['facecolor'],
                        edgecolor=style['edgecolor'],
                        linewidth=style['linewidth'])
    else:
        # update polygon
        hdl.set_xdata(X_rot)
        hdl.set_ydata(Y_rot)

    return hdl


def draw_unicycle(state, scale, hdl=None):
    """
    Draw a unicycle robot (rectangle body + two wheels).
    
    state: [x, y, theta]
    scale: scaling factor
    hdl: list of handles (for update), else None
    """
    if hdl is None:
        hdl = [None, None, None]

    x, y, theta = state
    length = 0.25

    # body size
    body_w = length * scale
    body_h = length * scale

    # wheel size
    wheel_w = 0.5 * length * scale
    wheel_h = 0.15 * length * scale
    wheel_shift = (body_h + wheel_h) / 2

    # styles
    body_style = {
        'facecolor': (255/255, 99/255, 0/255),   # orange
        'edgecolor': (70/255, 35/255, 10/255),   # light orange
        'linewidth': 0.1 * scale
    }

    wheel_style = {
        'facecolor': (70/255, 35/255, 10/255),   # brown
        'edgecolor': (70/255, 35/255, 10/255),
        'linewidth': 0.1 * scale
    }

    # main body
    param = [x, y, body_w, body_h, theta]
    hb = draw_rectangle(param, body_style, hdl[0])

    # wheels
    xw1 = x - wheel_shift * np.sin(theta)
    yw1 = y + wheel_shift * np.cos(theta)
    param = [xw1, yw1, wheel_w, wheel_h, theta]
    hw1 = draw_rectangle(param, wheel_style, hdl[1])

    xw2 = x + wheel_shift * np.sin(theta)
    yw2 = y - wheel_shift * np.cos(theta)
    param = [xw2, yw2, wheel_w, wheel_h, theta]
    hw2 = draw_rectangle(param, wheel_style, hdl[2])

    return [hb, hw1, hw2]

'''
# Example usage
if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # robot state: x=0, y=0, heading=30°
    state = [0, 0, np.pi/6]
    scale = 5
    hdl = draw_unicycle(state, scale)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()
'''