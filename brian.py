import helper as hf
import numpy as np

a = np.array([[10,100,10,100],
              [1,20,2,40],
              [1,2,30,4],
              [10,6,70,8]], float)

b = np.array([[100,2,30,4],
              [5,6,70,8],
              [1,200,3,4],
              [50,6,7,80]], float)

c = np.array([[1,2,30,4],
              [5,60,7,8],
              [1,2,30,4],
              [50,60,7,80]], float)

# a = np.ones((4,4))
# b = np.ones((4,4))
# c = np.ones((4,4))

# the regular attention
# softmax (a x b) x c = o

print("the regular attention")
o = hf.softmax(a@b)@c
regular_o1, regular_o2, regular_o3, regular_o4 = hf.partition(o, [2,2])
hf.print_matrix(regular_o1, "regular_o1")
hf.print_matrix(regular_o2, "regular_o2")
hf.print_matrix(regular_o3, "regular_o3")
hf.print_matrix(regular_o4, "regular_o4")
print("")

print("the distributed attention")
a1, a2, a3, a4 = hf.partition(a, [2,2])
b1, b2, b3, b4 = hf.partition(b, [2,2])
c1, c2, c3, c4 = hf.partition(c, [2,2])

p1 = a1@b1 + a2@b3
p2 = a1@b2 + a2@b4
p3 = a3@b1 + a4@b3
p4 = a3@b2 + a4@b4

m1 = hf.rowmax(p1)
e1 = np.exp(p1-m1)
l1 = hf.rowsum(e1)

m2 = hf.rowmax(p2)
e2 = np.exp(p2-m2)
l2 = hf.rowsum(e2)

m3 = hf.rowmax(p3)
e3 = np.exp(p3-m3)
l3 = hf.rowsum(e3)

m4 = hf.rowmax(p4)
e4 = np.exp(p4-m4)
l4 = hf.rowsum(e4)

s11 = hf.inverse(hf.diag(l1)) @ (e1@c1)
s12 = hf.inverse(hf.diag(l1)) @ (e1@c2)
s21 = hf.inverse(hf.diag(l2)) @ (e2@c3)
s22 = hf.inverse(hf.diag(l2)) @ (e2@c4)

s33 = hf.inverse(hf.diag(l3)) @ (e3@c1)
s34 = hf.inverse(hf.diag(l3)) @ (e3@c2)
s43 = hf.inverse(hf.diag(l4)) @ (e4@c3)
s44 = hf.inverse(hf.diag(l4)) @ (e4@c4)

M1 = np.full((len(m1), 1), 0)
for i in range(len(m1)):
    M1[i][0] = max(m1[i][0], m2[i][0])
L1 = (l1 * np.exp(m1-M1)) + (l2 * np.exp(m2-M1))

s11_new = hf.inverse(hf.diag(L1)) @ ((hf.diag(l1) * np.exp(m1-M1)) @ s11)
s12_new = hf.inverse(hf.diag(L1)) @ ((hf.diag(l1) * np.exp(m1-M1)) @ s12)

s21_new = hf.inverse(hf.diag(L1)) @ ((hf.diag(l2) * np.exp(m2-M1)) @ s21)
s22_new = hf.inverse(hf.diag(L1)) @ ((hf.diag(l2) * np.exp(m2-M1)) @ s22)

M2 = np.full((len(m3), 1), 0)
for i in range(len(m3)):
    M2[i][0] = max(m3[i][0], m4[i][0])
L2 = (l3 * np.exp(m3-M2)) + (l4 * np.exp(m4-M2))

s33_new = hf.inverse(hf.diag(L2)) @ ((hf.diag(l3) * np.exp(m3-M2)) @ s33)
s34_new = hf.inverse(hf.diag(L2)) @ ((hf.diag(l3) * np.exp(m3-M2)) @ s34)

s43_new = hf.inverse(hf.diag(L2)) @ ((hf.diag(l4) * np.exp(m4-M2)) @ s43)
s44_new = hf.inverse(hf.diag(L2)) @ ((hf.diag(l4) * np.exp(m4-M2)) @ s44)

distributed_o1 = s11_new + s21_new
distributed_o2 = s12_new + s22_new
distributed_o3 = s33_new + s43_new
distributed_o4 = s34_new + s44_new

hf.print_matrix(distributed_o1, "distributed_o1")
hf.print_matrix(distributed_o2, "distributed_o2")
hf.print_matrix(distributed_o3, "distributed_o3")
hf.print_matrix(distributed_o4, "distributed_o4")

# diff_o1 = regular_o1 - distributed_o1
# diff_o2 = regular_o2 - distributed_o2
# diff_o3 = regular_o3 - distributed_o3
# diff_o4 = regular_o4 - distributed_o4

# hf.print_matrix(diff_o1, "diff_o1")
# hf.print_matrix(diff_o2, "diff_o2")
# hf.print_matrix(diff_o3, "diff_o3")
# hf.print_matrix(diff_o4, "diff_o4")
