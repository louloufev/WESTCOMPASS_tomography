
f, ax = plt.subplots()

remove_all_points = 0
# RZ = np.array([R, Z])
# RZ = RZ.T
RZ = out
R = RZ[:, 0]
Z = RZ[:, 1]
while remove_all_points == 0:
    plt.plot(R, Z)
    points = utility_functions.draw_polygon(f, ax)
    print('wait', points)

    RZ = utility_functions.clean_points_inside_poly(points, RZ)
    R = RZ[:, 0]
    Z = RZ[:, 1]
    plt.plot(R, Z, 'r')
    remove_all_points = int(input('enter 1 if all points removed, 0 else'))
