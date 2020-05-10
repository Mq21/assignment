import numpy as np
import matplotlib.pyplot as plt

#***************************データの読み込み**************************************
while True:
    d_select=input('LLGMNデータ…「0」を入力, 筋電データ…「1」を入力: ')
    if d_select=='0':
        x_train = np.loadtxt('lea_sig.csv',delimiter=',')   #データの読み込み
        t_train = np.loadtxt('lea_T_sig.csv',delimiter=',')   #データの読み込み
        x_test = np.loadtxt('dis_sig.csv',delimiter=',')   #データの読み込み
        t_test = np.loadtxt('dis_T_sig.csv',delimiter=',')   #データの読み込み
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
#*********************************************************************************

#cl:クラス数,component:コンポーネント数,eta:学習率,iters_num:学習回数,batch_size:ミニバッチ学習のデータ数
cl,component,eta,iters_num,batch_size=t_train.shape[1],2,1,1000,100

print('データの非線形変換中...')

#学習データの非線形変換**********************************************
train_data_init=np.ones(1)
train_data_init=np.append(train_data_init,x_train[0])
for i in range(x_train.shape[1]):
    for j in range(i,x_train.shape[1]):
        train_data_init=np.append(train_data_init,x_train[0][i]*x_train[0][j])
train_data_init=train_data_init.reshape(1,train_data_init.size)
for i in range(1,x_train.shape[0]):
    train_data=np.ones(1)
    train_data=np.append(train_data,x_train[i])
    for j in range(x_train.shape[1]):
        for k in range(j,x_train.shape[1]):
            train_data=np.append(train_data,x_train[i][j]*x_train[i][k])
    train_data=train_data.reshape(1,train_data.size)
    train_data_init=np.append(train_data_init,train_data,axis=0)
x_train=train_data_init
#********************************************************************

#テストデータの非線形変換********************************************
test_data_init=np.ones(1)
test_data_init=np.append(test_data_init,x_test[0])
for i in range(x_test.shape[1]):
    for j in range(i,x_test.shape[1]):
        test_data_init=np.append(test_data_init,x_test[0][i]*x_test[0][j])
test_data_init=test_data_init.reshape(1,test_data_init.size)
for i in range(1,x_test.shape[0]):
    test_data=np.ones(1)
    test_data=np.append(test_data,x_test[i])
    for j in range(x_test.shape[1]):
        for k in range(j,x_test.shape[1]):
            test_data=np.append(test_data,x_test[i][j]*x_test[i][k])
    test_data=test_data.reshape(1,test_data.size)
    test_data_init=np.append(test_data_init,test_data,axis=0)
x_test=test_data_init
#********************************************************************

#重みの初期値
w=np.random.rand(x_train.shape[1],component*cl)
w[:,-1]=0

while True:
    key=input('逐次学習…「0」を入力, 一括学習…「1」を入力: ')

#逐次学習************************************************************
    if key=='0':
        print('逐次学習中...')
        delta=np.empty((x_train.shape[1],component*cl))
        for num in range(iters_num):
            batch_mask=np.random.choice(x_train.shape[0],batch_size)
            x_batch=x_train[batch_mask]
            t_batch=t_train[batch_mask]
            for i in range(x_batch.shape[0]):
                I=np.dot(x_batch[i],w)
                O=np.exp(I)/np.sum(np.exp(I))
                Y=np.empty(cl)
                for j in range(cl):
                    Y[j]=np.sum(O[component*j:component*(j+1)])
                    for k in range(x_batch.shape[1]):
                        delta[k,component*j:component*(j+1)]=(Y[j]-t_batch[i][j])*O[component*j:component*(j+1)]*x_batch[i][k]/Y[j]
                #重み更新
                w-=eta*delta
                w[:,-1]=0
        break
#********************************************************************

#一括学習************************************************************
    elif key=='1':
        print('一括学習中...')
        for num in range(iters_num):
            delta=np.zeros((x_train.shape[1],component*cl))
            batch_mask=np.random.choice(x_train.shape[0],batch_size)
            x_batch=x_train[batch_mask]
            t_batch=t_train[batch_mask]
            for i in range(x_batch.shape[0]):
                I=np.dot(x_batch[i],w)
                O=np.exp(I)/np.sum(np.exp(I))
                Y=np.empty(cl)
                for j in range(cl):
                    Y[j]=np.sum(O[component*j:component*(j+1)])
                    for k in range(x_batch.shape[1]):
                        delta[k,component*j:component*(j+1)]+=(Y[j]-t_batch[i][j])*O[component*j:component*(j+1)]*x_batch[i][k]/Y[j]         
            #重み更新
            w-=eta*delta
            w[:,-1]=0
        break
#********************************************************************

    else:
        print('無効な入力です')

result=np.empty((t_test.shape[0],t_test.shape[1]+1))
print('結果出力=逐次学習:result_one.csv，一括学習:result_all.csv')
print('1～{}列目:識別結果，{}列目:損失関数(0に近いほど正しく識別できている)'.format(t_test.shape[1],t_test.shape[1]+1))

#テスト**************************************************************
for i in range(x_test.shape[0]):
    I=np.dot(x_test[i],w)
    O=np.exp(I)/np.sum(np.exp(I))
    Y=np.empty(cl)
    acc=np.empty(cl)
    for j in range(cl):
        Y[j]=np.sum(O[component*j:component*(j+1)])
        acc[j]=t_test[i][j]*np.log(Y[j])
    acc_score=-np.sum(acc)
    result[i,:-1]=Y
    result[i,-1]=acc_score
if key=='0':
    np.savetxt('result_one.csv',result,delimiter=',')
elif key=='1':
    np.savetxt('result_all.csv',result,delimiter=',')
#********************************************************************
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