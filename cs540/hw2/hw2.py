import sys
import math
import string

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()

    # get a list of all the uppercase letters in the keys of X
    for letter in string.ascii_uppercase:
        X[letter] = 0

    with open (filename,encoding='utf-8') as f:
        for line in f:
            for character in line:
                if character.upper() in string.ascii_uppercase:
                    X[character.upper()] += 1
    return X


# Outputs

# input arguments
args = sys.argv
filename = args[1]
englishPrior = float(args[2])
spanishPrior = float(args[3])

# Q1
print("Q1")
dicty1 = shred(filename)
for key in sorted(dicty1.keys()):
    print(key, dicty1[key])
    
# Q2
print("Q2")
e, s = get_parameter_vectors()
x1LogE1 = dicty1["A"] * math.log(e[0])
x1LogS1 = dicty1["A"] * math.log(s[0])
print(float("{:.4f}".format(x1LogE1)))
print(float("{:.4f}".format(x1LogS1)))

# Q3
print("Q3")
if englishPrior == 0:
    logEnglishPrior = -100000000
elif spanishPrior == 0:
    logSpanishPrior = -100000000
else:
    logEnglishPrior = math.log(englishPrior)
    logSpanishPrior = math.log(spanishPrior)

# getting the english sum * log
sumXiEnglish = 0
counter = 0
for letter in dicty1:
    if e[counter] == 0:
        sumXiEnglish += dicty1[letter] * -100000000
    else:
        sumXiEnglish += dicty1[letter] * math.log(e[counter])
    counter += 1

# getting the spanish sum * log
sumXiSpanish = 0
counter = 0
for letter in dicty1:
    if s[counter] == 0:
        sumXiSpanish += dicty1[letter] * -100000000
    else:
        sumXiSpanish += dicty1[letter] * math.log(s[counter])
    counter += 1

# getting the final question values
fEnglish = logEnglishPrior + sumXiEnglish
fSpanish = logSpanishPrior + sumXiSpanish
print(float("{:.4f}".format(fEnglish)))
print(float("{:.4f}".format(fSpanish)))

# Q4
print("Q4")
pYEnglishX = 0
if fSpanish - fEnglish >= 100:
    pYEnglishX = 0
elif fSpanish - fEnglish <= -100:
    pYEnglishX = 1
else:
    pYEnglishX = 1/(1 + math.exp(fSpanish - fEnglish))
print(float("{:.4f}".format(pYEnglishX)))



