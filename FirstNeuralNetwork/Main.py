from Network2 import Network
import DataLoader
import Show

def main():

    train_data = DataLoader.load_train_data(10000)
    test_data = DataLoader.load_test_data(1000)
    network = Network([784, 30, 10])
    network.SGD(train_data, 30, 10, 3.0,test_data)

    myimage1 = DataLoader.load_image("images/1.png")
    myimage2 = DataLoader.load_image("images/2.png")
    myimage3 = DataLoader.load_image("images/3.png")
    myimage4 = DataLoader.load_image("images/4.png")
    myimage5 = DataLoader.load_image("images/5.png")
    myimage6 = DataLoader.load_image("images/6.png")
    myimage7 = DataLoader.load_image("images/7.png")
    myimage8 = DataLoader.load_image("images/8.png")
    myimage9 = DataLoader.load_image("images/9.png")


    print("1: ",network.myeval(myimage1))
    print("2: ",network.myeval(myimage2))
    print("3: ",network.myeval(myimage3))
    print("4: ",network.myeval(myimage4))
    print("5: ",network.myeval(myimage5))
    print("6: ",network.myeval(myimage6))
    print("7: ",network.myeval(myimage7))
    print("8: ",network.myeval(myimage8))
    print("9: ",network.myeval(myimage9))

    #Show.showDigitFromDB(train_data[95][0])
    #myimage = DataLoader.load_image("1.png")
    #Show.showDigitFromDB(myimage)


if __name__ == '__main__':
    main()