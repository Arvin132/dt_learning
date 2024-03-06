import heapq
from copy import deepcopy
import math
import numpy as np

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]
    
    
def information_gain(y: np.ndarray[int]) -> float:
    zeros = 0; ones = 0

    for val in y:
        if (val == 0):
            zeros += 1
        else:
            ones += 1
    p0 = zeros / len(y)
    p1 = ones / len(y)

    return -1 * (p0 * math.log2(p0) + p1 * math.log2(p1))


def split_data(inputs: np.ndarray, targets: np.ndarray, given: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        given inputs and targets, split the data in which the inputs contain the given int

        retVal:
            inpust_have, targets_have, inputs_dont_have, targets_dont_have
    """
    inputs_1 = []; targets_1 = []
    inputs_2 = []; targets_2 = []
    
    for idx in range(len(inputs)):
        if (np.isin(given, inputs[idx])):
            inputs_1.append(inputs[idx])
            targets_1.append(targets[idx])
        else:
            inputs_2.append(inputs[idx])
            targets_2.append(targets[idx])


    return np.array(inputs_1), np.array(targets_1), np.array(inputs_2), np.array(targets_2)

def information_gain_by_split(inputs: np.ndarray, targets: np.ndarray, given: int) -> float:
    zeros_have = 0; ones_have = 0
    zeros_no = 0; ones_no = 0

    for idx in range(len(inputs)):
        if (np.isin(given, inputs[idx])):
            if (targets[idx] == 0):
                zeros_have += 1
            else:
                ones_have += 1
        else:
            if (targets[idx] == 0):
                zeros_no += 1
            else:
                ones_no += 1

    E = information_gain(targets)
    p0_have = zeros_have / (zeros_have + ones_have)
    p1_have = ones_have / (zeros_have + ones_have)
    p0_no = zeros_no / (zeros_no + ones_no)
    p1_no = ones_no / (zeros_no + ones_no)
    E_split = ((zeros_have + ones_have) / len(targets)) * (-1 * (p0_have * math.log2(p0_have) + p1_have * math.log2(p1_have))) \
             + ((zeros_no + ones_no) / len(targets)) * (-1 * (p0_no * math.log2(p0_no) + p1_no * math.log2(p1_no)))

    return E - E_split 


        
class TreeNode:
    children: list = [] # list of TreeEdges 
    feature: int = 0
    inputs : np.ndarray[np.ndarray[int]]
    targets : np.ndarray[int]


    def __init__(self, inputs, targets) -> None:
        self.inputs = inputs
        self.targets = targets
    
    def add_child(self, child):
        self.children.append(child)
    


class TreeEdge:
    dest: TreeNode = None
    val: bool = False


class DesicionTree:
    root: TreeNode = None
    heap = PriorityQueue()
    depth = 0
    max_depth = 100

    def __init__(self, max_depth=100):
        self.max_depth=max_depth

    def fit(self, inputs: np.ndarray[np.ndarray[int]], targets: np.ndarray[int]):
        pass

    def predict(self, inputs: np.ndarray[np.ndarray[int]]) -> list[int]:
        return self(inputs)

    def __call__(self, inputs: np.ndarray[np.ndarray[int]]) -> list[int]:
        pass

