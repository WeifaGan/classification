import matplotlib.pyplot as plt

def result_visual(train_loss,val_loss,train_acc,val_acc):
# plot函数作图
    plt.plot(train_loss,label='train loss')  
    plt.plot(val_loss,label='val loss')  
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("iteration times")
    plt.title("Loss Result")
    plt.show() 

    plt.plot(train_acc,label='train acc')  
    plt.plot(val_acc,label='val acc')  
    plt.legend()
    plt.ylabel("acc")
    plt.xlabel("iteration times")
    plt.title("acc Result")
    plt.show() 
    print("draw end!")



if __name__ == "__main__":
    train_loss = [11,21,31,4,5,6]
    train_acc = [12,24,35,41,51,62]
    val_loss = [10,2.1,3.1,4,5,6]
    val_acc   = [1.4,2.2,3.5,4.6,5.4,6.6]
    result_visual(train_loss,val_loss,train_acc,val_acc)
