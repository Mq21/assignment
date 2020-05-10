clear
n=4096;         %データ数
dt=0.005;       %サンプリング間隔
t=((1:n)-1)*dt;
f=t/dt/dt/n;    
fs=1/dt;        %サンプリング周波数
dt2=0.0025;       %サンプリング間隔
u=((1:n)-1)*dt2;
fc_high = 10;   %ハイパス用
fc_low = 70;    %ローパス用

%遮断周波数やサンプル数などは上のものを使ってください

%sin波の作成と合成波の作成
y = sin(2*pi*5*t) + sin(2*pi*50*t) + sin(2*pi*80*t);

%フィルターの作成.
%フィルタの次数は2でやりました.
[b_l,a_l] = butter(2,fc_low/(fs/2),'low');
[b_h,a_h] = butter(2,fc_high/(fs/2),'high');
[b_p,a_p] = butter(2,[fc_high fc_low]/(fs/2),'bandpass');

% ダウンサンプリング（サンプル数を減らす）
t_d = downsample(t,2);
y_d = downsample(y,2);


% アップサンプリング：3次スプライン補間（サンプル数を増やす）
y_u = spline(t,y,u);

% グラフ作成
figure(1)
subplot(2,1,1)
plot(t,y)
xlim([0,0.3])
title('3周波数の合成波形')

subplot(2,1,2)
y1 = fft(y);
y2 = abs(y1);
plot(f,y2)
title('合成波形のFFT')

figure(2)
subplot(2,2,1)
y_l=filter(b_l,a_l,y);
plot(t,y_l)
xlim([0,0.3])
title('ローパス通過後の合成波形')

subplot(2,2,2)
y_h=filter(b_h,a_h,y);
plot(t,y_h)
xlim([0,0.3])
title('ハイパス通過後の合成波形')

subplot(2,2,3)
y_p=filter(b_p,a_p,y);
plot(t,y_p)
xlim([0,0.3])
title('バンドパス通過後の合成波形')

figure(3)
subplot(2,2,1)
y1_l = fft(y_l);
y2_l = abs(y1_l);
plot(f,y2_l)
title('ローパス後のFFT')

subplot(2,2,2)
y1_h = fft(y_h);
y2_h = abs(y1_h);
plot(f,y2_h)
title('ハイパス後のFFT')

subplot(2,2,3)
y1_p = fft(y_p);
y2_p = abs(y1_p);
plot(f,y2_p)
title('バンドパス後のFFT')

figure(4);
subplot(2,2,1)
stem(t,y,'b');
xlim([0,0.3])
title('元信号')

subplot(2,2,2)
stem(t_d,y_d,'MarkerFaceColor','r');
xlim([0,0.3])
title('ダウンサンプリングされた信号')

subplot(2,2,3)
stem(t_d,y_d,'MarkerFaceColor','r');
hold on;
stem(u,y_u,'b');
xlim([0 0.3])
title('アップサンプリングされた信号')

