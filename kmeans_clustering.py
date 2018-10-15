import random
import math

def dotProduct(d1, d2):
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())


def increment(d1, scale, d2):
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run.
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''

    # Select K random centers
    centers = random.sample(examples, K)
    assignments = [None] * len(examples)
    totalCost = [None] * len(examples)
    precalc_ex = [[]] * len(examples)

    # Refactor out the ex ** 2 value
    for i in range(len(examples)):
        precalc_ex[i] = dotProduct(examples[i], examples[i])

    for n in range(maxIters):

        prev_assignments = assignments[:]

        precalc_centers = [[]] * len(centers)

        # Refactor out the c ** 2 value
        for j in range(len(centers)):
            precalc_centers[j] = dotProduct(centers[j], centers[j])

        # update assignments
        for i in range(len(examples)):
            minval = float('inf')

            for j in range(len(centers)):

                # Instead of doing an expensive multiplication multiple times, the refactoring allows a simple look up
                sum = (-2*dotProduct(examples[i], centers[j]) + precalc_ex[i] + precalc_centers[j]) ** 0.5

                # assign the element to the closest center
                if sum < minval:
                    minval = sum
                    assignments[i] = j
                    totalCost[i] = sum ** 2

        if assignments == prev_assignments:
            break

        prev_centers = centers[:]

        # update centers as the average of all its assigned values
        for j in range(len(centers)):
            temp = {}
            total = 0

            for i in range(len(assignments)):
                if assignments[i] == j:
                    increment(temp, 1.0, examples[i])
                    total += 1
            for f, v in temp.items():
                temp[f] = v/total
            centers[j] = temp

        if prev_centers == centers:
            break

    return centers, assignments, math.fsum(totalCost)


# Sample test
x1 = {0:0, 1:0}
x2 = {0:0, 1:1}
x3 = {0:0, 1:2}
x4 = {0:0, 1:3}
x5 = {0:0, 1:4}
x6 = {0:0, 1:5}
examples = [x1, x2, x3, x4, x5, x6]
centers, assignments, totalCost = kmeans(examples, 2, maxIters=10)

print centers
print assignments
print totalCost