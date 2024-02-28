from copy import deepcopy


"""
    Dataset used to store each word and their respective index
"""
class WordDataset:

    words = {}

    def __init__(self):
        pass

    def read_file(self, file_path: str):
        """
            given path to the file of the text input, reads and loads the data
        """
        
        with open(file_path, mode="r") as f:
            input = f.readline(); idx = 1
            while(input != None and len(input) != 0):
                self.words[idx] = input
                input = f.readline()
                idx += 1
            
    def __getitem__(self, given: int) -> str:
        return self.words[given]


class InputDataset:

    docs : list[list[int]] = []

    def __init__(self):
        pass

    def read_file(self, input_file_path) -> str:
        """
            given path to the file of the text input, reads and loads the data
        """
        

        with open(input_file_path, mode="r") as f:
            input = f.readline()
            prev_idx = 0
            while(input != None and len(input) != 0):
                input = input.strip()
                elems = input.split()
                idx = int(elems[0])
                word = int(elems[1])
                while(prev_idx <= idx):
                    self.docs.append([])
                    prev_idx += 1

                self.docs[idx - 1].append(word)
                input = f.readline()

                

    def get_vals(self) -> list[list[int]]:
        return deepcopy(self.docs)
    

class TargetDataset:

    labels : list[int] = []

    def __init__(self):
        pass

    def read_file(self, target_file_path) -> str:
        """
            given path to the file of the text input, reads and loads the data
        """
        with open(target_file_path, mode="r") as f:
            input = f.readline()
            while(input != None and len(input) != 0):
                input = input.strip()
                self.labels.append(int(input))
                input = f.readline()

    def get_vals(self) -> list[int]:
        return deepcopy(self.labels)
    


def test():
    wds = WordDataset()
    wds.read_file("dataset/words.txt")
    print(wds.words[2])


    ids = InputDataset()
    ids.read_file("dataset/trainData.txt")
    print(ids.get_vals()[10: 20])

    lds = TargetDataset()
    lds.read_file("dataset/trainLabel.txt")
    print(lds.get_vals()[0: 10])

