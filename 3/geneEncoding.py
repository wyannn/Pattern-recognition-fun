# 0.0 coding:utf-8 0.0
import random


def geneEncoding(pop_size, chrom_length):
    pop = []
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    for i in range(pop_size):
        if sum(pop[i]) == 0:
            pop[i] = [1, 1, 1, 1, 1, 1]
        if sum(pop[i]) < 2:
            pop[i] = [1, 1, 0, 0, 0, 0]

    return pop


if __name__ == '__main__':
    pop_size = 50		# 种群数量
    chrom_length = 6		# 染色体长度
    pop = geneEncoding(pop_size, chrom_length)
    print(pop)
    print(len(pop))
