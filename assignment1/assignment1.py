import math
from typing import Self

class Vec:
    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements =[]
        else:
            elements = list(src)

            for x in elements: 
                if not isinstance (x, (int, float)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")
            self.elements = elements

    #Display the vector
    def __repr__(self) -> str:
        return repr(self.elements)

    #Return the number of elements in the vector
    def __len__(self) -> int:
        return len(self.elements)

    #Calculate the mean of the vector
    def mean(self) -> float:
        if len(self.elements) == 0:
            raise ValueError("cannot calculate mean of an empty vector")
        return sum(self.elements) / len(self.elements)

    #Return the demean vector:
    def demean(self) -> Self:
        mean = self.mean()

        return Vec([x - mean for x in self.elements])

    #Calculate the standard deviation

    def std(self) -> float:
        demeaned = self.demean()
        squared_deviations = [x * x for x in demeaned.elements]

        return math.sqrt(sum(squared_deviations) / len(self.elements))
    
v1 = Vec([1, 5, 6])

print(v1)
print(v1.mean())
print(v1.demean())
print(v1.std())  