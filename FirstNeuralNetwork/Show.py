import matplotlib.pyplot as plt

def showDigitFromDB(image):
    array = []
    i = 0;
    newrow = []
    for item in image:
        if i < 28:
            newrow.append(1 - item[0])
            i += 1
        else:
            array.append(newrow)
            newrow = []
            i = 0
    h = 14
    for j in range(0, 14):
        for i in range(0, 27):
            temp = array[i][j]
            array[i][j] = array[i][j + h]
            array[i][j + h] = temp
    plt.imshow(array, cmap=plt.get_cmap('gray'))
    plt.show()