import numpy as np
import matplotlib.pyplot as plt

#***************************データの読み込み**************************************
while True:
    d_select=input('NNデータ…「0」を入力, 筋電データ…「1」を入力: ')
    if d_select=='0':
        a = np.loadtxt('data.csv',delimiter=',')
        x_train=a[:-2,:-1]
        t_train=a[:-2,-1]
        t_train=t_train.reshape(t_train.size,1)
        x_test=a[:,:-1]
        t_test=a[:,-1]
        t_test=t_test.reshape(t_test.size,1)
        break
    elif d_select=='1':
        a = np.loadtxt('experiment2.csv',delimiter=',')   #データの読み込み
        x_train = a[:,:4]                       #学習データ
        t_train = a[:,4:]                        #学習データの教師データ
        b = np.loadtxt('experiment3.csv',delimiter=',')
        x_test = b[:,:4]                          #テストデータ
        t_test = b[:,4:]                           #正解データ
        break
    else:
        print('無効な入力です')
train_data_shape = np.shape(x_train)       #学習データのサイズ
test_data_shape = np.shape(x_test)         #テストデータのサイズ
#*********************************************************************************

#eta:学習率,iters_num:学習回数,batch_size:ミニバッチ学習のデータ数
eta,iters_num,batch_size=1,1000,100

#*******************************Classの定義***************************************
class Sigmoid():                                      #シグモイド関数
    def forward(self , array):
        buf = array.copy()
        mask = (array < -100)
        buf[mask] = -100
        mask = (array > 100)
        buf[mask] = 100
        
        return 1 / (1 + np.exp(-buf))
    
    def backward(self,array,delta):

        return array * (1-array) * delta

class Loss():                                         #損失関数
    def forward(self,output,teacher):

        return (output-teacher)**2

    def backward(self,output,teacher):

        return (output-teacher)*2

class Input_Layer():                                  #入力層のパラメータ設定
    def __init__(self,number):
        self.number = number
        self.outlayer = np.zeros((self.number))
        
class Hidden_Layer():                                 #隠れ層のパラメータ設定
    def __init__(self,input,number):
        self.number = number
        self.inlayer = np.zeros((self.number))
        self.outlayer = np.zeros((self.number))
        self.weight  = np.abs(np.random.rand(self.number , input.number + 1))
        for i in range(self.weight.shape[-2]):
            ave = np.average(self.weight[i,:])
            SD = np.std(self.weight[i,:])
            if SD == 0:
                SD = 0.00000001
            self.weight[i,:] = (self.weight[i,:] - ave) / SD

class Output_Layer():                                 #出力層のパラメータ設定
    def __init__(self,hidden,number):
        self.number = number
        self.inlayer = np.zeros((self.number))
        self.outlayer = np.zeros((self.number))
        self.weight  = np.abs(np.random.rand(self.number , hidden.number + 1))
        for i in range(self.weight.shape[-2]):
            ave = np.average(self.weight[i,:])
            SD = np.std(self.weight[i,:])
            if SD == 0:
                SD = 0.00000001
            self.weight[i,:] = (self.weight[i,:] - ave) / SD

class Full_Conect_Calc():                             #計算
    def forward(self,input,output,func):              #順伝搬
        for i in range(output.number):
            output.inlayer[i] = np.dot(input.outlayer , output.weight[i,0:input.number]) 
            output.inlayer[i] -= output.weight[i , input.number]
        output.outlayer = func.forward(output.inlayer)

    def backward(self,leftlayer,rightlayer,delta,func):    #逆伝搬
        delta = func.backward(rightlayer.outlayer , delta)
        d = delta.copy()
        delta = np.dot(delta , rightlayer.weight[: , 0 : leftlayer.number])
        #重みの更新
        rightlayer.weight[: , 0 : leftlayer.number] -= eta*np.dot(d.reshape(-1,1) , leftlayer.outlayer.reshape(1 , leftlayer.number))

        return delta

loss = Loss() 
sigmoid = Sigmoid()
FC = Full_Conect_Calc()

#ここから!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
input = Input_Layer(x_train.shape[1])
hidden = Hidden_Layer(input,4)                        #隠れ層のノード数指定!!!!!
output = Output_Layer(hidden,t_train.shape[1])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ここまで

#*********************************************************************************

#**********************************学習*******************************************
#error = 200
print('学習中...')
for num in range(iters_num):
    batch_mask=np.random.choice(train_data_shape[0],batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]
    for i in range(x_batch.shape[0]):
        input.outlayer = x_batch[i]                 #学習データ入力
        teacher_data = t_batch[i]                     #学習データの教師データ入力

        #順伝搬
        #ここから!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        FC.forward(input,hidden,sigmoid)
        FC.forward(hidden,output,sigmoid)  
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ここまで

        #error += np.sum(loss.forward(output.outlayer,teacher_data))  #損失関数計算

        #誤差逆伝搬
        delta = loss.backward(output.outlayer,teacher_data)
        #ここから!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        delta = FC.backward( hidden , output , delta , sigmoid)
        delta = FC.backward( input , hidden , delta , sigmoid)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ここまで

    #error = error / train_data_shape[0]
#*********************************************************************************

#**********************************テスト*****************************************
result=np.empty((t_test.shape[0],t_test.shape[1]+1))
print('結果出力=result.csv')
print('1～{}列目:識別結果，{}列目:損失関数(0に近いほど正しく識別できている)'.format(t_test.shape[1],t_test.shape[1]+1))
for i in range(test_data_shape[0]):
    input.outlayer = x_test[i]                      #テストデータ入力

    #順伝搬
    #ここから!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    FC.forward(input,hidden,sigmoid)
    FC.forward(hidden,output,sigmoid)   
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ここまで

    #損失関数
    acc=np.empty(t_test.shape[1])
    for j in range(t_test.shape[1]):
        acc[j]=t_test[i][j]*np.log(output.outlayer[j])
    acc_score=-np.sum(acc)
    #結果をファイルに書き込み
    result[i,:-1]=output.outlayer
    result[i,-1]=acc_score
np.savetxt('result.csv',result,delimiter=',')
#*********************************************************************************
t=np.arange(t_test.shape[0])
plt.figure(1)
plt.plot(t+1,t_test)
if d_select=='0':
    plt.savefig('test.png')
elif d_select=='1':
    plt.savefig('test2.png')
plt.figure(2)
plt.plot(t+1,result[:,:-1])
if d_select=='0':
    plt.savefig('estimation.png')
elif d_select=='1':
    plt.savefig('estimation2.png')
plt.show()