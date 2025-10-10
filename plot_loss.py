import matplotlib.pyplot as plt

# data1 = []
# with open("loss_result_1_1000.txt", 'r') as f:
#     items = f.readlines()
#     for item in items:
#         data1.append(float(item.split()[2]))
#
# data5 = []
# with open("loss_result_5_1000.txt", 'r') as f:
#     items = f.readlines()
#     for item in items:
#         data5.append(float(item.split()[2]))
#
# data10 = []
# with open("loss_result_10_1000.txt", 'r') as f:
#     items = f.readlines()
#     for item in items:
#         data10.append(float(item.split()[2]))
#
# data100 = []
# with open("loss_result_100_1000.txt", 'r') as f:
#     items = f.readlines()
#     for item in items:
#         data100.append(float(item.split()[2]))

data1002 = []
with open("loss_weight_100.txt", 'r') as f:
    items = f.readlines()
    for item in items:
        data1002.append(float(item.split()[2]))


data10002 = []
with open("loss_weight_100.txt", 'r') as f:
    items = f.readlines()
    for item in items:
        data10002.append(float(item.split()[2]))
# plt.plot(data1)
# plt.plot(data5)
# plt.plot(data10)
# plt.plot(data100)
plt.plot(data1002)
plt.plot(data10002)
plt.show()