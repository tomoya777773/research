import numpy as np
import matplotlib.pyplot as plt




# a = np.load("../data/gpis_error_200.npy")
# b = np.load("../data/mtgpis_error_200_21.npy")
# c = np.load("../data/mtgpis_error_200_23.npy")
# d = np.load("../data/mtgpis_error_200_25.npy")
# e = np.load("../data/mtgpis_error_200_30.npy")
# f = np.load("../data/mtgpis_error_200_50.npy")
# g = np.load("../data/mtgpis_error_200_cos.npy")

# x = np.arange(len(b))
# print a
# plt.plot(x, a, linewidth=4, color="red")
# plt.plot(x, b, linewidth=4, color="blue")
# plt.plot(x, c, linewidth=4, color="yellow")
# plt.plot(x, d, linewidth=4, color="pink")
# plt.plot(x, e, linewidth=4, color="black")
# plt.plot(x, f, linewidth=4, color="navy")
# plt.plot(x, g, linewidth=4, color="orange")

# plt.show()

# a = np.load("../data/gpis_var_ave_200.npy")
# b = np.load("../data/mtgpis_var_ave_200_21.npy")
# c = np.load("../data/mtgpis_var_ave_200_23.npy")
# d = np.load("../data/mtgpis_var_ave_200_25.npy")
# e = np.load("../data/mtgpis_var_ave_200_30.npy")
# f = np.load("../data/mtgpis_var_ave_200_50.npy")
# g = np.load("../data/mtgpis_var_ave_200_cos.npy")

# x = np.arange(len(b))

# plt.plot(x, a, linewidth=4, color="red")
# plt.plot(x, b, linewidth=4, color="blue")
# plt.plot(x, c, linewidth=4, color="yellow")
# plt.plot(x, d, linewidth=4, color="pink")
# plt.plot(x, e, linewidth=4, color="black")
# plt.plot(x, f, linewidth=4, color="navy")
# plt.plot(x, g, linewidth=4, color="orange")

# plt.show()

a = np.load("../data/gpis/gpis_error.npy")
b = np.load("../data/mtgpis/ellipse21/error.npy")
c = np.load("../data/mtgpis/ellipse25/error.npy")
d = np.load("../data/mtgpis/ellipse30/error.npy")
e = np.load("../data/mtgpis/ellipse80/error.npy")
f = np.load("../data/mtgpis/ellipse100/error.npy")
g = np.load("../data/mtgpis/ellipse1000/error.npy")

x = np.arange(len(b))
# print a
plt.plot(x, a, linewidth=4, color="red")
plt.plot(x, b, linewidth=4, color="blue")
plt.plot(x, c, linewidth=4, color="yellow")
plt.plot(x, d, linewidth=4, color="pink")
# plt.plot(x, e, linewidth=4, color="black")
plt.plot(x, f, linewidth=4, color="navy")
# plt.plot(x, g, linewidth=4, color="orange")

plt.show()

a = np.load("../data/gpis/gpis_var_ave.npy")
b = np.load("../data/mtgpis/ellipse21/var.npy")
c = np.load("../data/mtgpis/ellipse25/var.npy")
d = np.load("../data/mtgpis/ellipse30/var.npy")
e = np.load("../data/mtgpis/ellipse80/var.npy")
f = np.load("../data/mtgpis/ellipse100/var.npy")
# g = np.load("../data/mtgpis_error_200_cos.npy")

x = np.arange(len(b))
# print a
plt.plot(x, a, linewidth=4, color="red")
plt.plot(x, b, linewidth=4, color="blue")
plt.plot(x, c, linewidth=4, color="yellow")
plt.plot(x, d, linewidth=4, color="pink")
# plt.plot(x, e, linewidth=4, color="black")
plt.plot(x, f, linewidth=4, color="navy")
# plt.plot(x, g, linewidth=4, color="orange")

plt.show()