
import pandas as pd
import plost
import base64
import numpy as np
from scipy.integrate import simpson
from numpy import trapz
from PIL import Image
import tkinter



#data = 3260000 + ((78*60*60*24*365)



#data = (3.0842*(10**13)) + (78*60*60*24*365)
ausbreitung = (78*60*60*24*365)
#Mg = (3.0842*(10**13))

n = 10000
sum_of_numbers = 0
i = 1
while i <= n:
    sum_of_numbers += ausbreitung
    #print(sum_of_numbers)
    i += 1
#result = "Summe von 1 bis " + str(n) + ": " + str((3.0842*(10**13)) + sum_of_numbers)
result = str((3.0842*(10**13)) + sum_of_numbers)
result2 = ((3.0842*(10**13)) + sum_of_numbers)
print(result)


#print(Mg - float(result2))


data = 30842000000000 + (78*60*60*24*365)

print(data)