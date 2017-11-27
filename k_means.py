import csv
import copy
import numpy as np

filename = ['eye.csv', 'finger.csv', 'foot.csv', 'hand.csv', 'leg.csv']

def save_z_means(filename):
    classZ = []
    for i in range(5):
        z_value = []
        fileread = open('class_z_value/'+filename[i], 'rb')
        csvreader = csv.reader(fileread, delimiter=',')
        for row in csvreader:
            oneStroke = []
            for item in row:
                oneStroke.append(float(item))
            z_value.append(copy.deepcopy(oneStroke))
        allZ = np.array(z_value)
        classZ.append(np.mean(allZ, axis = 0))
        fileread.close()
    np.save('z_means.npy', classZ)
    
def accuracy(filename, npyfile):
    z_means = np.load(npyfile)
    total = 0
    correct = 0
    for i in range(5):
        fileread = open('class_z_value/'+filename[i], 'rb')
        csvreader = csv.reader(fileread, delimiter=',')
        for row in csvreader:
            oneStroke = []
            for item in row:
                oneStroke.append(float(item))
            testData = np.array(oneStroke)
            diff = np.zeros(5)
            for j in range(5):
                diff[j] = np.linalg.norm(testData - z_means[j])
            if np.argmin(diff) == i:
                correct += 1
            total += 1
            #if total % 200 == 0:
            #    print(float(correct) / total)
    print(correct, total, float(correct)/total)
                
accuracy(filename, 'z_means.npy')
            
#print(classZ)
'''
    if i < 109063:
        element = []
        for item in row:
            try:
                element.append(float(item))
            except:
                pass
        classZ.append(element)
        i += 1
    elif i == 109063:
        z_value.append(copy.deepcopy(classZ))
        classZ = []
        i += 1
        continue
    elif i < 258279:
        element = []
        for item in row:
            element.append(float(item))
        classZ.append(element)
        i += 1
    elif i == 258279:
        z_value.append(copy.deepcopy(classZ))
        classZ = []
        i += 1
        continue
    elif i < 436803:
        element = []
        for item in row:
            element.append(float(item))
        classZ.append(element)
        i += 1
    elif i == 436803:
        z_value.append(copy.deepcopy(classZ))
        classZ = []
        i += 1
        continue
    elif i < 697605:
        element = []
        for item in row:
            element.append(float(item))
        classZ.append(element)
        i += 1
    elif i == 697605:
        z_value.append(copy.deepcopy(classZ))
        classZ = []
        i += 1
        continue
    elif i < 799721:
        element = []
        for item in row:
            element.append(float(item))
        classZ.append(element)
        i += 1
    elif i == 799721:
        z_value.append(copy.deepcopy(classZ))
        classZ = []
        i += 1
        continue
z_means = []
for i in range(len(z_value)):
    allZ = np.array(z_value[i])
    z_means.append(np.mean(allZ, axis = 0))
print(z_means)
'''
        

