def readLog(filename):
    f = open(filename)
    l = []
    single_epoch=''
    maxacc=0
    maxepoch=0
    for line in f.readlines():
        if '****' in line:
            single_epoch = line.strip().split(' ')[-2]
        if 'eval mIoU:' in line:
            single_accuracy = line.strip().split(' ')[-1]
            # print(single_mIoU)
            l.append((single_accuracy,single_epoch))
            if float(single_accuracy) > maxacc:
                maxacc=float(single_accuracy)
                maxepoch=int(single_epoch)

    print("Current is {} epoch.".format(l[-1][-1]))
    print(l)
    for i in range(int(l[-1][-1])+1):
        print("{}".format(l[i][-2]))
    print("{} epoch get the highest accuracy {}".format(str(maxepoch),str(maxacc)))

if __name__ == '__main__':
    # readLog('bao.txt')
    readLog('cmysemantic_graph_based_3frames_2.txt')
