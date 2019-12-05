import gym.envs.toy_text.frozen_lake as frozenLake
import copy
import random

directions = ['R', 'L', 'U', 'D']
map = frozenLake.generate_random_map()
print(map)
QVals = {}
for i in range(len(map)):
    QVals[i] = {}
    for j in range(len(map)):
        if map[i][j] == 'S':
            start_i = i
            start_j = j
        elif map[i][j] == 'G':
            goal_i = i
            goal_j = j
        QVals[i][j] = {}
        for direction in directions:
            QVals[i][j][direction] = 0

def existInMap(new_i,new_j):
    if len(map) > max(new_i,new_j) and min(new_j,new_i) >= 0:
        return True
    else:
        return False

def isTerminal(curr_i, curr_j):
    if existInMap(curr_i,curr_j) == False:
        return None
    if map[curr_i][curr_j] in ['H','G']:
        return True
    else:
        return False

def RFunc(curr_i, curr_j):
    reward = 1
    if goal_i == curr_i and goal_j == curr_j:
        return reward
    else:
        return 0


def getMaxQ(i, j):
    maxGrade = -1
    maxAction = ''
    for direction in directions:
        if maxGrade < QVals[i][j][direction]:
            maxGrade = QVals[i][j][direction]
            maxAction = direction
    return maxGrade,maxAction


learning_rate = 0.9
discount_factor = 0.9
decay_rate = 0.995
epsilon = 1
episode_number = 0
episodes_limit = 5000
steps_limit = 100


def sample_action(i, j):
    if random.uniform(0,1) > epsilon:
        maxGrade, maxAction = getMaxQ(i, j)
        return maxAction
    else:
        return directions[random.randrange(4)]

while episode_number < episodes_limit:
    if episode_number in [500,2000]:
        print(QVals)
    current_i = start_i
    current_j = start_j
    new_QVals = copy.deepcopy(QVals)
    step_number = 0
    while step_number < steps_limit:
        current_action = sample_action(current_i,current_j)
        target = None
        if current_action == 'R':
            new_i = current_i
            new_j = current_j + 1
        elif current_action == 'L':
            new_i = current_i
            new_j = current_j - 1
        elif current_action == 'U':
            new_i = current_i - 1
            new_j = current_j
        elif current_action == 'D':
            new_i = current_i + 1
            new_j = current_j
        if existInMap(new_i,new_j) == False:
            step_number += 1
            continue
        if isTerminal(new_i,new_j):
            target = RFunc(new_i,new_j)
            new_QVals[current_i][current_j][current_action] = QVals[current_i][current_j][current_action] * (
                        1 - learning_rate) + learning_rate * target
            break
        ##
        maxGrade, maxAction = getMaxQ(new_i, new_j)
        target = RFunc(new_i,new_j) + discount_factor * maxGrade
        new_QVals[current_i][current_j][current_action] = QVals[current_i][current_j][current_action]*(1-learning_rate)+learning_rate*target
        current_i = new_i
        current_j = new_j
        step_number += 1
    QVals = new_QVals
    episode_number += 1
    epsilon *= decay_rate
print(QVals)
