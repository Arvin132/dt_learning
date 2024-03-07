import heapq
from copy import deepcopy
import math
import numpy as np
from Dataset import WordDataset


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]
    
    def is_empty(self) -> bool:
        return (len(self._queue) == 0)
       
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
            mask = inputs[idx] != given
            inputs_1.append(inputs[idx][mask])
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
    feature = 0         # feature to do the split on
    ig_val = 0          # information gain value
    inputs : np.ndarray[np.ndarray[int]]
    targets : np.ndarray[int]
    prediction = -1

    def __init__(self, inputs, targets) -> None:
        self.inputs = inputs
        self.targets = targets
    
    def add_child(self, child):
        self.children.append(child)

    def find_optimal_split(self) -> tuple[int, float]:
        """
            finds the optimal feature to do the split, and then stores that in feature and the score
            on ig_val fields of itself

            retVal:
                feature, split_score
        """

        words = set()
        for word in self.inputs.flatten():
            words.add(word)

        
        self.feature = words[0]
        self.ig_val = information_gain_by_split(self.inputs, self.targets, words[0])
        for word in words:
            val = information_gain_by_split(self.inputs, self.targets, word)
            if (val > self.ig_val):
                self.feature = word
                self.ig_val = val
        
        return self.feature, self.ig_val
    
    def predict(self, given_input) -> int:
        """
            given an input will return the predicted value
        """
        # if internal node
        if (len(self.children) != 0):
            for child in self.children:
                if(np.isin(given_input, self.feature) == child.val):
                    return child.dest.predict(given_input)
                
        #if leaf node        
        if (self.prediction == -1):
            p = np.count_nonzero(self.targets) / len(self.targets)
            if (p >= 0.5):
                self.prediction = 1
            else:
                self.prediction = 0
        return self.prediction


  
class TreeEdge:
    dest: TreeNode = None
    val: bool = False

    def __init__(self, dest, val) -> None:
        self.dest = dest
        self.val = val

class DesicionTree:
    root: TreeNode = None
    heap = PriorityQueue()
    max_size: int
    size = 0

    def __init__(self, max_size=100):
        self.max_size=max_size

    def fit(self, inputs: np.ndarray[np.ndarray[int]], targets: np.ndarray[int], words: WordDataset):
        if (self.root != None):
            raise RuntimeError(" supposed to use fit method only once after object init")
        
        self.root = TreeNode(deepcopy(inputs), deepcopy(targets))
        self.depth = 1
        feature, split_val = self.root.find_optimal_split()
        self.heap.push(self.root, split_val)

        while(self.heap.is_empty() == False and self.size <= self.max_size):
            node: TreeNode = self.heap.pop()
            # create the new nodes
            x_left, y_left, x_right, y_right = split_data(node.inputs, node.targets, node.feature)
            left_node = TreeNode(x_left, y_left)
            right_node = TreeNode(x_right, y_right)
            # push the new nodes into the heap
            l_f, l_val = left_node.find_optimal_split()
            r_f, r_val = right_node.find_optimal_split()
            self.heap.push(left_node, l_val)
            self.heap.push(right_node, r_val)
            # add new nodes as children
            left_branch = TreeEdge(left_node, False)
            right_branch = TreeEdge(right_node, True)
            node.add_child(left_branch)
            node.add_child(right_branch)
            # increase size
            self.size += 2
            
    def predict(self, inputs: np.ndarray[np.ndarray[int]]) -> list[int]:
        return self(inputs)

    def __call__(self, inputs: np.ndarray[np.ndarray[int]]) -> list[int]:
        return [self.root.predict(x) for x in inputs]
