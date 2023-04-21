import matplotlib.pyplot as plt
f = open("/home/vinayak/CS7020/UBN/lr=0.1_BN_Train.txt", "r")
epoch_1e1_train = f.read().strip('][').split(', ')
for i in range(len(epoch_1e1_train)):
    epoch_1e1_train[i] = float(epoch_1e1_train[i])
    epoch_1e1_train[i] = epoch_1e1_train[i] * 100

f = open("/home/vinayak/CS7020/UBN/lr=0.003_BN_Train.txt", "r")
epoch_3e3_train = f.read().strip('][').split(', ')
for i in range(len(epoch_3e3_train)):
    epoch_3e3_train[i] = float(epoch_3e3_train[i])
    epoch_3e3_train[i] = epoch_3e3_train[i] * 100
    
f = open("/home/vinayak/CS7020/UBN/lr=0.0001_BN_Train.txt", "r")
epoch_1e4_train = f.read().strip('][').split(', ')
for i in range(len(epoch_1e4_train)):
    epoch_1e4_train[i] = float(epoch_1e4_train[i])
    epoch_1e4_train[i] = epoch_1e4_train[i] * 100

f = open("/home/vinayak/CS7020/UBN/lr=0.0001_WoBN_Train.txt", "r")
epoch_wo1e4_train = f.read().strip('][').split(', ')
for i in range(len(epoch_wo1e4_train)):
    epoch_wo1e4_train[i] = float(epoch_wo1e4_train[i])
    epoch_wo1e4_train[i] = epoch_wo1e4_train[i] * 100
    
f = open("/home/vinayak/CS7020/UBN/lr=0.1_BN_Test.txt", "r")
epoch_1e1_test = f.read().strip('][').split(', ')
for i in range(len(epoch_1e1_test)):
    epoch_1e1_test[i] = float(epoch_1e1_test[i])
    epoch_1e1_test[i] = epoch_1e1_test[i] * 100

f = open("/home/vinayak/CS7020/UBN/lr=0.003_BN_Test.txt", "r")
epoch_3e3_test = f.read().strip('][').split(', ')
for i in range(len(epoch_3e3_test)):
    epoch_3e3_test[i] = float(epoch_3e3_test[i])
    epoch_3e3_test[i] = epoch_3e3_test[i] * 100
    
f = open("/home/vinayak/CS7020/UBN/lr=0.0001_BN_Test.txt", "r")
epoch_1e4_test = f.read().strip('][').split(', ')
for i in range(len(epoch_1e4_test)):
    epoch_1e4_test[i] = float(epoch_1e4_test[i])
    epoch_1e4_test[i] = epoch_1e4_test[i] * 100
    
f = open("/home/vinayak/CS7020/UBN/lr=0.0001_WoBN_Test.txt", "r")
epoch_wo1e4_test = f.read().strip('][').split(', ')
for i in range(len(epoch_wo1e4_test)):
    epoch_wo1e4_test[i] = float(epoch_wo1e4_test[i])
    epoch_wo1e4_test[i] = epoch_wo1e4_test[i] * 100

# plt.plot(epoch_1e1_train, label='lr = 0.1 (with BN)')
# plt.plot(epoch_3e3_train, label='lr = 0.003 (with BN)')
# plt.plot(epoch_1e4_train, label='lr = 0.0001 (with BN)')
# plt.plot(epoch_wo1e4_train, label='lr = 0.0001 (without BN)')
# plt.legend()
# # plt.show()
# plt.savefig("acc_train.png")

plt.plot(epoch_1e1_test, label='lr = 0.1 (with BN)')
plt.plot(epoch_3e3_test, label='lr = 0.003 (with BN)')
plt.plot(epoch_1e4_test, label='lr = 0.0001 (with BN)')
plt.plot(epoch_wo1e4_test, label='lr = 0.0001 (without BN)')
plt.legend()
# plt.show()
plt.savefig("acc_test.png")