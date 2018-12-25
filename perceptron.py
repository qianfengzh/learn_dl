#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from __future__ import print_function
from functools import reduce


class VectorOp(object):
    """
    Implement vector calculation
    """
    @staticmethod
    def dot(x, y):
        """
        计算向量内积(对应相乘再相加)
        :param x:
        :param y:
        :return:
        """
        return reduce(lambda a, b: a + b, VectorOp.element_multiply(x, y), 0.0)

    @staticmethod
    def element_multiply(x, y):
        """
        向量元素对应相乘
        :param x:
        :param y:
        :return:
        """
        return list(map(lambda x_y: x_y[0] * x_y[1], zip(x, y)))

    @staticmethod
    def element_add(x, y):
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))

    @staticmethod
    def scala_multiply(v, s):
        """
        向量元素与标量相乘
        :param v:
        :param s:
        :return:
        """
        return map(lambda e: e * s, v)


class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0] * input_num
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vect):
        return self.activator(VectorOp.dot(input_vect, self.weights) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
        return self

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = VectorOp.element_add(self.weights, VectorOp.scala_multiply(input_vec, rate * delta))
        self.bias += rate * delta


def func_activator(x):
    return 1 if x > 0 else 0


def get_training_dataset():
    """
    基于and真值表构建训练数据
    """
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():
    """
    使用and真值表训练感知器
    """
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(2, func_activator)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return p


if __name__ == '__main__':
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perception)
    # 测试
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))




