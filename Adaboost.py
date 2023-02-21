import pandas as pd
import math
from copy import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dg_deep = 1 # 决策树层数
classifier_num = 11 # 基分类器的数量
is_draw=1   #是否画图

divide_line = []  # 划分点
divide_sx = []  # 划分属性
divide_line_num=[]   #划分点总数

# is_minerror=1   #是否采用最小化误差代替最大化信息增益

# class node:
# 获取列表的第二个元素(int)
def takeSecond(elem):
    return elem[1]


class Root:  # 决策树的树根
    num = 0  # 属性数量
    numbers = 0  # 数据个数
    attributes = []  # 属性值
    attributes_list = []  # 全部属性值列表
    is_leaf = 0  # 0为非叶子节点，1为是，2为否
    next = []  # 节点列表
    data_weight = []  # 数据权重
    is_root = 1  # 是否是树根
    deep = 0  # 当前节点的深度

    # next_num = []   #孩子列表
    # next_len = 0    #当前节点列表中的节点数量
    # divide_list = [0, 0, 0.0, 0]  # 划分表    记录划分属性，第一个值为序号，第二个为是否为连续值，第三个为连续值划分点

    # 第四个为离散值的属性序号  或者  0表示小于该划分点，1表示大于该划分点

    def __init__(self, node):
        self.deep = node.deep
        self.is_root = 1
        self.next = []
        # self.next_num=[]
        # self.next_len=0
        self.is_leaf = 0
        self.num = node.num
        self.numbers = node.numbers
        self.attributes = node.attributes
        self.attributes_list = node.attributes_list
        self.data_weight = node.data_weight

    def show_data(self):
        print("属性数量:" + str(self.num))
        print("数据个数:" + str(self.numbers))
        for i in range(self.numbers):
            for j in range(self.num):
                print(self.attributes[j][i], end=" ")
            print('')
        print("属性值:")

        for i in range(self.num):
            print("第%d个属性的取值：" % (i + 1), end=' ')
            print(self.attributes_list[i])
        # print("划分表：", end='')
        # print(self.divide_list)

    def my_predict(self, data):
        if self.is_leaf == 1:
            return '是'
        elif self.is_leaf == 2:
            return '否'

        is_continue = self.next[0].divide_list[1]  # 是否是连续值
        sx_num = self.next[0].divide_list[0]  # 属性序号
        continue_divide = self.next[0].divide_list[2]  # 连续值划分点

        if is_continue:  # 连续值
            if data[sx_num] < continue_divide:
                return self.next[0].my_predict(data)  # 小于划分点
            else:
                return self.next[1].my_predict(data)  # 大于划分点
        else:
            new_data = []
            for i in range(len(data)):  # 建立一条新数据，除去该属性
                if i != sx_num:
                    new_data.append(data[i])

            for i in range(len(self.attributes_list[sx_num])):
                if data[sx_num] == self.attributes_list[sx_num][i]:
                    # print("new_data:", end=' ')
                    # print(new_data)
                    return self.next[i].my_predict(new_data)


class Node(Root):  # 节点类
    divide_list = []

    def __init__(self, node, init_list):  # init_list [0, 0, 0.0,0]  记录划分属性，第一个值为序号，第二个为是否为连续值，第三个为连续值划分点
        # 四个为离散值的属性序号  或者  0表示小于该划分点，1表示大于该划分点
        self.deep = node.deep
        self.divide_list = init_list
        self.is_root = 0
        self.next = []
        self.data_weight = []  # deepcopy(node.data_weight)
        self.is_leaf = 0
        if init_list[1] == 1:  # 连续值
            self.num = node.num
            self.numbers = 0
            self.attributes_list = node.attributes_list
            # self.divide_list = init_list
            self.attributes = [[] for x in range(node.num)]
            # print(self.attributes,init_list[0])
            if init_list[3] == 0:  # 小于划分点
                for i in range(node.numbers):
                    if node.attributes[init_list[0]][i] < init_list[2]:

                        self.data_weight.append(node.data_weight[i])

                        self.numbers += 1
                        for j in range(node.num):
                            self.attributes[j].append(node.attributes[j][i])
            else:
                for i in range(node.numbers):
                    if node.attributes[init_list[0]][i] > init_list[2]:

                        self.data_weight.append(node.data_weight[i])

                        self.numbers += 1
                        for j in range(node.num):
                            self.attributes[j].append(node.attributes[j][i])
        else:  # 离散值
            self.num = node.num - 1
            self.numbers = 0
            # self.divide_list = init_list
            self.attributes = [[] for x in range(node.num - 1)]
            self.attributes_list = [[] for x in range(node.num - 1)]

            temp_num = 0
            for i in range(node.num):  # 构建属性取值表
                if i != init_list[0]:
                    self.attributes_list[temp_num] = node.attributes_list[i]
                    temp_num += 1

            for i in range(node.numbers):  # 构建数据表
                if node.attributes[init_list[0]][i] == node.attributes_list[init_list[0]][init_list[3]]:
                    self.data_weight.append(node.data_weight[i])
                    self.numbers += 1

                    temp_num = 0
                    for j in range(node.num):
                        if j != init_list[0]:
                            self.attributes[temp_num].append(node.attributes[j][i])
                            temp_num += 1


class watermalon:
    num = 0  # 属性数量
    numbers = 0  # 数据个数
    attributes = []  # 属性值
    attributes_list = []  # 全部属性值列表
    is_leaf = 0  # 0为非叶子节点，1为是，2为否
    next = []
    data_weight = []
    deep = 0

    # attributes_p = []  # 各个属性取值的好坏瓜概率

    def __init__(self, num):
        self.deep = 0
        self.num = num
        self.numbers = 0
        for i in range(num):
            self.attributes.append([])
            self.attributes_list.append([])
            # self.attributes_p.append([])

    def add_data(self, *args):
        # print(args)
        for i in range(self.num):
            self.attributes[i].append(args[i])
            if not isinstance(args[i], str):
                continue
            elif args[i] not in self.attributes_list[i]:
                self.attributes_list[i].append(args[i])
                # self.attributes_p.append([])

        self.numbers += 1

    def show_data(self):
        print("属性数量:" + str(self.num))
        print("数据个数:" + str(self.numbers))
        for i in range(self.numbers):
            for j in range(self.num):
                print(self.attributes[j][i], end=" ")
            print('')
        print("属性值:")

        for i in range(self.num):
            print("第%d个属性的取值：" % (i + 1), end=' ')
            print(self.attributes_list[i])


def math_log(digit):  # 返回log值，若输入为0则返回0
    if digit == 0:
        return 0
    else:
        return math.log2(digit)


def TreeGenerate(data, A):
    global now_deep, dg_deep

    # print("递归开始")
    # data.show_data()
    # new_node = deepcopy(data)
    # new_node.is_leaf = 1
    if data.numbers == 0:
        return deepcopy(data)
    data.is_leaf = 1
    for i in range(data.numbers):  # D中样本是否属于同一类别
        if data.attributes[data.num - 1][i] != data.attributes[data.num - 1][0]:
            data.is_leaf = 0
            break
    if data.is_leaf != 0:
        if data.attributes[data.num - 1][0] == '是':
            data.is_leaf = 1
        else:
            data.is_leaf = 2
        # print("这是一个叶子")
        return deepcopy(data)

    if data.deep >= dg_deep:  # 超过了预设递归深度
        #print(len(data.attributes))
        #print(len(data.data_weight))
        yb_good = 0
        yb_bad = 0
        for i in range(data.numbers):
            if data.attributes[data.num - 1][i] == "是":
                yb_good += data.data_weight[i]
            else:
                yb_bad += data.data_weight[i]
        if yb_good > yb_bad:
            data.is_leaf = 1
        else:
            data.is_leaf = 2
        return deepcopy(data)
    else:
        data.deep += 1
    # for i in A:

    # 计算信息增益
    p_good = 0
    p_bad = 0
    all_num = 0.0

    weight_num = 0
    for j in data.attributes[data.num - 1]:
        if j == '是':
            p_good += data.data_weight[weight_num]
        else:
            p_bad += data.data_weight[weight_num]
        all_num += data.data_weight[weight_num]
        weight_num += 1
    p_good /= all_num  # 好瓜占比
    p_bad /= all_num  # 坏瓜占比
    ent_d = -(p_good * math_log(p_good) + p_bad * math_log(p_bad))  # 信息熵

    max = 0.0  # 信息增益最大值
    max_num = -1
    final_num = 0.0  # 连续值的划分点
    max_final_num = 0.0  # 最佳划分点
    for i in A:
        if data.attributes_list[i]:  # 离散值
            temp = [[0, 0] for x in range(len(data.attributes_list[i]))]  # 第一个为是，第二个为否
            for t in range(len(data.attributes_list[i])):
                for j in range(data.numbers):  # 对每一个数据遍历
                    if data.attributes_list[i][t] == data.attributes[i][j]:  # 该数据为该属性的某个取值
                        if data.attributes[data.num - 1][j] == "是":
                            temp[t][0] += data.data_weight[j]   #权重累加
                        else:
                            temp[t][1] += data.data_weight[j]
            sum = 0
            for my_list in temp:
                num_good = my_list[0]
                num_bad = my_list[1]

                if num_good != 0 and num_bad != 0:
                    p_good = num_good / float(num_good + num_bad)
                    p_bad = num_bad / float(num_good + num_bad)
                else:
                    p_good = 0
                    p_bad = 0

                if p_good == 0:  # 防止出现log2(0)
                    p_good = 1
                if p_bad == 0:
                    p_bad = 1

                ent = -(p_good * math.log2(p_good) + p_bad * math.log2(p_bad))
                sum += ((num_good + num_bad) / float(data.numbers)) * ent

            final_ent = ent_d - sum  # 最终信息增益
        else:  # 连续值
            temp_data = []  # 划分点候选值
            for t in range(data.numbers):
                temp_data.append(
                    (data.attributes[data.num - 1][t] == '是', data.attributes[i][t],
                     data.data_weight[t]))  # 第一个元素0为否，1为是，第二个元素为连续值,第三个元素为权重
            temp_data.sort(key=takeSecond)
            # print(temp_data)
            final_ent = 0
            final_num = 0.0  # 划分点数值
            # print(data.data_weight)
            for t in range(len(temp_data) - 1):  # 遍历划分点
                front_good = 0
                front_bad = 0
                back_good = 0
                back_bad = 0
                for count in range(len(temp_data)):

                    if temp_data[count][0]:  # 是
                        if count <= t:
                            front_good += temp_data[count][2]
                        else:
                            back_good += temp_data[count][2]
                    else:  # 否
                        if count <= t:
                            front_bad += temp_data[count][2]
                        else:
                            back_bad += temp_data[count][2]
                front_num = front_good + front_bad  # 前半部分的数据条数
                back_num = back_good + back_bad  # 后半部分的数据条数

                if front_good + front_bad == 0:  # 前后两部分的好瓜占比
                    front_good_p = 0
                else:
                    front_good_p = front_good / (front_good + front_bad)
                if back_good + back_bad == 0:
                    back_good_p = 0
                else:
                    back_good_p = back_good / (back_good + back_bad)

                if front_good_p == 0 or front_good_p == 1:
                    front_ent = 0
                else:
                    front_ent = -front_good_p * math.log2(front_good_p) - (1 - front_good_p) * math.log2(
                        1 - front_good_p)
                if back_good_p == 0 or back_good_p == 1:
                    back_ent = 0
                else:
                    back_ent = -back_good_p * math.log2(back_good_p) - (1 - back_good_p) * math.log2(1 - back_good_p)
                now_ent = ent_d - (front_num / (front_num + back_num)) * front_ent - (
                        back_num / (front_num + back_num)) * back_ent
                # print("序号:%d,now_ent:%.4f,划分点：%.3f" % (i,now_ent, (temp_data[t][1] + temp_data[t + 1][1]) / 2))
                if now_ent > final_ent:  # 取信息增益最大值
                    final_ent = now_ent
                    final_num = (temp_data[t][1] + temp_data[t + 1][1]) / 2  # 上述过程中，选取各个值作为划分点，此句将这些值转换成两个值的平均值
        # print("信息增益:%f" % final_ent)
        # if not data.attributes_list[i]:
        # print("划分点值:%f" % final_num)
        if final_ent > max:
            max = final_ent
            max_num = i
            if not data.attributes_list[i]:
                max_final_num = final_num  # 划分点
    print("信息增益最大:%f,属性序号:%d，属性名:%s,划分点：%f" % (max, max_num,data.attributes_list[max_num], max_final_num))
    ent_flag = [0, 0, 0.0, 0]  # 记录划分属性，第一个值为序号，第二个为是否为连续值，第三个为连续值划分点
    # 第四个为离散值的属性序号  或者  0表示小于该划分点，1表示大于该划分点
    ent_flag[0] = max_num

    node_next = []  # 临时变量

    if data.attributes_list[max_num]:  # 离散值
        ent_flag[1] = 0

        # print("--------------------属性取值个数:%d-------------------------" % len(data.attributes_list[max_num]))
        for tt in range(len(data.attributes_list[max_num])):
            # print("-----第%d个子树-----" % (tt+1))
            # print(data.attributes_list[max_num][tt])
            temp = Node(deepcopy(data), deepcopy(ent_flag))
            data.next.append(TreeGenerate(deepcopy(temp), deepcopy(A[:len(A) - 1])))
            # data.next_num.append(Root.next_len)
            # Root.next_len+=1
            # print("子树个数：%d" % len(data.next))
            # temp.show_data()
            ent_flag[3] += 1
    else:
        # print("-----1-----" )
        global divide_line
        #if dg_deep == 2:  # 在单层决策树时引入划分点画图
        if is_draw:
            divide_line.append(max_final_num)
            divide_sx.append(ent_flag[0])
        ent_flag[1] = 1
        ent_flag[2] = max_final_num
        ent_flag[3] = 0  # 小于划分值
        temp = Node(deepcopy(data), deepcopy(ent_flag))
        data.next.append(TreeGenerate(deepcopy(temp), deepcopy(A[:len(A) - 1])))
        # data.next_num.append(Root.next_len)
        # Root.next_len+=1
        # temp.show_data()
        ent_flag[3] = 1  # 大于划分值
        temp = Node(deepcopy(data), deepcopy(ent_flag))
        data.next.append(TreeGenerate(deepcopy(temp), deepcopy(A[:len(A) - 1])))
        # data.next_num.append(Root.next_len)
        # Root.next_len+=1
        # temp.show_data()

    # data.divide_list = deepcopy(ent_flag)  # 该节点的划分表，用于测试样本预测
    return deepcopy(data)


def adaboost_predict(data, classifier_list, classifier_weight, result):  # 结果预测，用于中途测试
    ture_num = 0
    false_num = 0
    count = 0
    for one_data in data:
        putout = 0
        for i in range(len(classifier_list)):

            if classifier_list[i].my_predict(one_data) == "是":
                putout += classifier_weight[i]
            else:
                putout += -classifier_weight[i]
        if putout > 0:
            putout = 1
        else:
            putout = -1

        if putout == result[count]:
            ture_num += 1
        else:
            false_num += 1
        count += 1
    print("中途测试错误率:%f" % (float(false_num) / (ture_num + false_num)))
    return float(false_num) / (ture_num + false_num)


def adaboost_train(data, base, epoch):  # 训练adaboost分类器,data为实际输出，1或-1,base为西瓜数据类
    base_weight = 1.0 / base.numbers  # 基础权重
    data_weight = [base_weight for x in range(base.numbers)]  # 样本权重
    classifier_list = []  # 分类器列表
    classifier_weight = [0 for x in range(epoch)]  # 分类器权重
    error_list = []
    now_base = deepcopy(base)  # 当前训练数据类
    for i in range(epoch):  # 训练每个分类器
        # now_base.show_data()
        now_base.data_weight = deepcopy(data_weight)
        now_base.deep = 0
        # print("实际数据权重:")
        # print(now_base.data_weight)
        classifier = deepcopy(Root(now_base))
        classifier = TreeGenerate(classifier, np.linspace(0, now_base.num - 1, now_base.num - 1, endpoint=False).astype(
            np.int))  # 建立决策树
        # print("数据个数:%d" % classifier.numbers)
        classification_results = [0 for x in range(base.numbers)]  # 分类结果，正例为1，反例为-1
        num_yes = 0
        num_no = 0
        error_rate = 0.0
        new_data_list = []  # 完整的数据集，用于adaboost_predict
        for t in range(base.numbers):
            new_data = []
            for ii in range(base.num - 1):
                new_data.append(base.attributes[ii][t])
            new_data_list.append(new_data)
            # 判断是否是第一个学习器
            # if i == 0:
            result = classifier.my_predict(new_data)
            # else:
            # result = adaboost_predict(new_data, classifier_list, classifier_weight)

            if base.attributes[base.num - 1][t] == result:  # 分类正确
                num_yes += data_weight[t]
                # print("第%d个分类正确,权重%f"% (t,data_weight[t]))
            else:
                num_no += data_weight[t]
                error_rate += data_weight[t]
                # print("第%d个分类错误,权重%f"% (t,data_weight[t]))
            if result == "是":
                classification_results[t] = 1  # data_weight[t]
            else:
                classification_results[t] = -1  # data_weight[t]
        # print(classification_results)
        error_rate = num_no / float(num_yes + num_no)
        #if error_rate>0.45:
        #    error_rate=math.fabs(error_rate-0.5)
        print("error_rate:%f" % error_rate)
        if error_rate > 0.5:
            print("训练错误,第%d个分类器的错误率大于0.5" % (i + 1))
            break

        if error_rate == 0:
            classifier_weight[i] = 1  # 若分类完全正确，权重为max
            #print("当前分类器已达到100%正确率！")
            #break
        else:
            classifier_weight[i] = 0.5 * math.log((1 - error_rate) / error_rate)  # 当前分类器权重

        classifier_list.append(deepcopy(classifier))  # 将新的分类器添加到分类器列表中
        # 重新分配样本权重
        zt = 0  # 规范化因子
        min_weight = 1
        for j in range(base.numbers):
            zt += data_weight[j] * math.exp(-classifier_weight[i] * data[j] * classification_results[j])
        for j in range(base.numbers):
            data_weight[j] = data_weight[j] / zt * math.exp(-classifier_weight[i] * data[j] * classification_results[j])
            if data_weight[j] < min_weight:
                min_weight = data_weight[j]

        error_list.append(adaboost_predict(new_data_list, classifier_list, classifier_weight, data))
        divide_line_num.append(len(divide_line))
        # print("样本权重:",end=" ")
        # print(data_weight)
        # 对样本进行基于权重的过采样
        """
        now_base = deepcopy(base)  # 当前训练数据类

        for j in range(base.numbers):
            temp_num = (data_weight[j] - min_weight) / (min_weight / 1000)
            for jj in range(int(temp_num + 0.5)):  # 对temp_num四舍五入
                now_base.numbers += 1
                for tt in range(now_base.num):
                    now_base.attributes[tt].append(now_base.attributes[tt][j])
        """
    draw(error_list, classifier_num)
    return classifier_list, classifier_weight


def adaboost_test(data, classifier_list, classifier_weight):
    ture_num = 0
    false_num = 0
    for one_data in data:
        putout = 0
        for i in range(len(classifier_list)):

            if classifier_list[i].my_predict(one_data) == "是":
                putout += classifier_weight[i]
            else:
                putout += -classifier_weight[i]

        print("当前值：%c,实际数据%f,预测值%f" % (one_data[-1], one_data[-2], putout), end=',')
        if putout > 0:
            putout = "是"
        else:
            putout = "否"

        if putout == one_data[-1]:
            ture_num += 1
            print("预测正确")
        else:
            false_num += 1
            print("预测错误")
    print("正确率：%f" % (float(ture_num) / (ture_num + false_num)))

    return


def draw(err, m):
    try:
        plt.title('基学习器个数 = %d时, err = 0' % (np.where(np.array(err) == 0)[0][0] + 1))
    except IndexError:
        plt.title('err != 0')
    # plt.plot(range(1, len(lost) + 1), lost, 'o--', markersize=2, label='lost')
    plt.plot(range(1, len(err) + 1), err, 'o--', markersize=2, label='err')
    # plt.plot([1, len(lost)], [1 / m, 1 / m], 'k-.', linewidth=0.3, label='1/m')
    plt.xlabel('基学习器个数')
    plt.ylabel('错误率')
    plt.legend()
    plt.show()


def draw_predict(one_data, classifier_list, classifier_weight):  # 结果预测，用于中途测试
    putout = 0.0
    for i in range(len(classifier_list)):
        if classifier_list[i].my_predict(one_data) == "是":
            putout += classifier_weight[i]
        else:
            putout += -classifier_weight[i]
    return putout

def predict(H, feature1, feature2,classifier_list,classifier_weight):
    # 预测结果,仅feature1和feature2两个特征,feature1和feature2同维度
    pre = np.zeros(feature1.shape)
    one_data=[0,0]
    for i in range(100):
       for j in range(100):
           one_data[0]=feature1[i][j]
           one_data[1]=feature2[i][j]
           pre[i][j]=draw_predict(one_data,classifier_list,classifier_weight)
    """
    for h in H:
        alpha, feature, point = h   #alpha 分类器权重    feature 属性序号    point 划分点
        pre += alpha * (((feature1 * (feature == 0) + feature2 * (feature == 1)) <= point) * 2 - 1)
    """
    return np.sign(pre)


def draw_pic(X, Y, H, T ,classifier,classifier_weight):
    x1min, x1max = X[:, 0].min(), X[:, 0].max()
    x2min, x2max = X[:, 1].min(), X[:, 1].max()
    x1_l, x1_h = x1min - (x1max - x1min) * 0.2, x1max + (x1max - x1min) * 0.2
    x2_l, x2_h = x2min - (x2max - x2min) * 0.2, x2max + (x2max - x2min) * 0.2
    x1, x2 = np.linspace(x1_l, x1_h, 100), np.linspace(x2_l, x2_h, 100)
    X1, X2 = np.meshgrid(x1, x2)  # 矩阵网格
    #print(X2.shape)
    for t in [3, 5, 11, 19]:
        if t > T:
            break
        plt.title('%d个基学习器,决策树层数:%d' % (t,dg_deep))
        plt.xlabel("密度")
        plt.ylabel("含糖量")
        plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='+', c='k', s=80, label='好瓜')
        plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='_', c='k', s=80, label='坏瓜')
        plt.legend()
        # 画基学习器划分边界
        for i in range(divide_line_num[t-1]):
            feature, point = H[i, 1:]
            if feature == 0:
                plt.plot([point, point], [x2_l, x2_h], 'k', linewidth=1)
            else:
                plt.plot([x1_l, x1_h], [point, point], 'k', linewidth=1)
        # 画集成学习器划分边界
        print(H)
        Y_pre = predict(H[:t], X1, X2,classifier[:t],classifier_weight[:t])
        plt.contour(X1, X2, Y_pre, colors='r', linewidths=3, levels=0)  # 网格矩阵分成-1/1两类，画出水平线
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv('watermelon3_0_Ch.csv', index_col=0)
    w = watermalon(data.shape[1])
    for i in range(data.shape[0]):  # 数据读取
        temp_data = []
        for j in data.columns:
            temp_data.append(data[j][i + 1])
        w.add_data(*tuple(temp_data))
    w.show_data()

    # tree_root = Root(w)

    real_y = []
    for i in w.attributes[w.num - 1]:
        if i == "是":
            real_y.append(1)
        else:
            real_y.append(-1)

    # tree_root = TreeGenerate(tree_root, np.linspace(0, w.num - 1, w.num - 1, endpoint=False).astype(np.int))  # 建立决策树

    classifier_list, classifier_weight = adaboost_train(real_y, w, classifier_num)

    test_data = []
    for i in range(w.numbers):
        temp = []
        for j in range(w.num):
            temp.append(w.attributes[j][i])
        test_data.append(temp)

    # for i in range(len(classifier_list)):
    # print(classifier_list[i].data_weight)

    adaboost_test(test_data, classifier_list, classifier_weight)

    print(len(classifier_list), classifier_weight)

    if is_draw:
        # 画图
        draw_data = []
        for i in range(w.numbers):
            draw_data.append([w.attributes[0][i], w.attributes[1][i]])
        draw_data = np.array(draw_data)

        classifier_draw = []
        for i in range(divide_line_num[-1]):
            if i<classifier_num:
                classifier_draw.append([classifier_weight[i], divide_sx[i], divide_line[i]])
            else:
                classifier_draw.append([0, divide_sx[i], divide_line[i]])
        draw_pic(np.array(draw_data), np.array(real_y), np.array(classifier_draw), classifier_num,classifier_list,classifier_weight)

    # print(tree_root.next[0].next)
    # print(tree_root.divide_list)
    # w.priori_probability()
    # w.test_data("青绿","蜷缩","浊响","清晰","凹陷","硬滑",0.697,0.46)
    # w.show_data()
    # print(data)
    # print(data.columns)
