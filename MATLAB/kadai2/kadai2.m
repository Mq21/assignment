clear
n1=41401;         %実験1のデータ数
n2=46380;         %実験2のデータ数
n3=45290;         %実験3のデータ数
n4=45626;         %実験4のデータ数
n5=18564;
n6=23409;
n7=21254;
n8=23659;
dt=0.001;       %サンプリング間隔
t1=((1:n1)-1)*dt;
t2=((1:n2)-1)*dt;
t3=((1:n3)-1)*dt;
t4=((1:n4)-1)*dt;
t5=((1:n5)-1)*dt;
t6=((1:n6)-1)*dt;
t7=((1:n7)-1)*dt;
t8=((1:n8)-1)*dt;
fs=1/dt;        %サンプリング周波数

%筋電データの読み込み
A = load('13_36_55check.txt');  %実験1
y_a=A(:,2:5);
B = load('13_37_39check.txt');  %実験2
y_b=B(:,2:5);
C = load('13_38_27check.txt');  %実験3
y_c=C(:,2:5);
D = load('13_39_15check.txt');  %実験4
y_d=D(:,2:5);

%バンドパス
[b_p,a_p] = butter(2,[1.5 100]/(fs/2),'bandpass');
yp_a=filter(b_p,a_p,y_a);
yp_b=filter(b_p,a_p,y_b);
yp_c=filter(b_p,a_p,y_c);
yp_d=filter(b_p,a_p,y_d);

%全波整流
y2_a=abs(yp_a);
y2_b=abs(yp_b);
y2_c=abs(yp_c);
y2_d=abs(yp_d);

%ローパス
[b_l,a_l] = butter(2,1/(fs/2),'low');
yl_a=filter(b_l,a_l,y2_a);
yl_b=filter(b_l,a_l,y2_b);
yl_c=filter(b_l,a_l,y2_c);
yl_d=filter(b_l,a_l,y2_d);

%オフセット除去
yo_a(:,1)=yl_a(:,1)-0.0054444858582070261;
yo_a(:,2)=yl_a(:,2)-0.0083672618404349083;
yo_a(:,3)=yl_a(:,3)-0.0017889026320701926;
yo_a(:,4)=yl_a(:,4)-0.004543593989920173;
yo_b(:,1)=yl_b(:,1)-0.0054444858582070261;
yo_b(:,2)=yl_b(:,2)-0.0083672618404349083;
yo_b(:,3)=yl_b(:,3)-0.0017889026320701926;
yo_b(:,4)=yl_b(:,4)-0.004543593989920173;
yo_c(:,1)=yl_c(:,1)-0.0054444858582070261;
yo_c(:,2)=yl_c(:,2)-0.0083672618404349083;
yo_c(:,3)=yl_c(:,3)-0.0017889026320701926;
yo_c(:,4)=yl_c(:,4)-0.004543593989920173;
yo_d(:,1)=yl_d(:,1)-0.0054444858582070261;
yo_d(:,2)=yl_d(:,2)-0.0083672618404349083;
yo_d(:,3)=yl_d(:,3)-0.0017889026320701926;
yo_d(:,4)=yl_d(:,4)-0.004543593989920173;

%各筋力ごとの最大値で正規化
ys_a(:,1)=yo_a(:,1)/0.029586337422735077;
ys_a(:,2)=yo_a(:,2)/0.059626142026547951;
ys_a(:,3)=yo_a(:,3)/0.063928339809234291;
ys_a(:,4)=yo_a(:,4)/0.062394027980363205;
ys_b(:,1)=yo_b(:,1)/0.029586337422735077;
ys_b(:,2)=yo_b(:,2)/0.059626142026547951;
ys_b(:,3)=yo_b(:,3)/0.063928339809234291;
ys_b(:,4)=yo_b(:,4)/0.062394027980363205;
ys_c(:,1)=yo_c(:,1)/0.029586337422735077;
ys_c(:,2)=yo_c(:,2)/0.059626142026547951;
ys_c(:,3)=yo_c(:,3)/0.063928339809234291;
ys_c(:,4)=yo_c(:,4)/0.062394027980363205;
ys_d(:,1)=yo_d(:,1)/0.029586337422735077;
ys_d(:,2)=yo_d(:,2)/0.059626142026547951;
ys_d(:,3)=yo_d(:,3)/0.063928339809234291;
ys_d(:,4)=yo_d(:,4)/0.062394027980363205;

%ノイズ除去+閾値超えデータ抜き出し
M_a=max(ys_a,[],2);
S_a=sum(ys_a,2);
cnt_a=1;
for i=1:41401
    if M_a(i)<=1&&S_a(i)>0.5
        y3_a(cnt_a,:)=ys_a(i,:);
        cnt_a=cnt_a+1;
    end
end
M_b=max(ys_b,[],2);
S_b=sum(ys_b,2);
cnt_b=1;
for i=1:46380
    if M_b(i)<=1&&S_b(i)>0.5
        y3_b(cnt_b,:)=ys_b(i,:);
        cnt_b=cnt_b+1;
    end
end
M_c=max(ys_c,[],2);
S_c=sum(ys_c,2);
cnt_c=1;
for i=1:45290
    if M_c(i)<=1&&S_c(i)>0.5
        y3_c(cnt_c,:)=ys_c(i,:);
        cnt_c=cnt_c+1;
    end
end
M_d=max(ys_d,[],2);
S_d=sum(ys_d,2);
cnt_d=1;
for i=1:45626
    if M_d(i)<=1&&S_d(i)>0.5
        y3_d(cnt_d,:)=ys_d(i,:);
        cnt_d=cnt_d+1;
    end
end

% 各チャンネルの総和が各時間において合計１となるように調整
ysum_a=sum(y3_a,2);
ysum_b=sum(y3_b,2);
ysum_c=sum(y3_c,2);
ysum_d=sum(y3_d,2);
for i = 1:18564
    y3_a(i,:)=y3_a(i,:)/ysum_a(i);
end
for i = 1:23409
    y3_b(i,:)=y3_b(i,:)/ysum_b(i);
end
for i = 1:21254
    y3_c(i,:)=y3_c(i,:)/ysum_c(i);
end
for i = 1:23659
    y3_d(i,:)=y3_d(i,:)/ysum_d(i);
end

%プロット
subplot(2,2,1)
plot(1:n5,y3_a)
subplot(2,2,2)
plot(1:n6,y3_b)
subplot(2,2,3)
plot(1:n7,y3_c)
subplot(2,2,4)
plot(1:n8,y3_d)

%データ書き込み
y3_a(:,5:8)=0;
y3_b(:,5:8)=0;
y3_c(:,5:8)=0;
y3_d(:,5:8)=0;
for i=1:1845
    y3_a(i,5)=1;
end
for i=1846:6950
    y3_a(i,6)=1;
end
for i=6951:11764
    y3_a(i,7)=1;
end
for i=11765:18564
    y3_a(i,8)=1;
end
for i=1:4565
    y3_b(i,5)=1;
end
for i=4566:11502
    y3_b(i,6)=1;
end
for i=11503:15975
    y3_b(i,7)=1;
end
for i=15976:23409
    y3_b(i,8)=1;
end
for i=1:3624
    y3_c(i,5)=1;
end
for i=3625:9033
    y3_c(i,6)=1;
end
for i=9034:13503
    y3_c(i,7)=1;
end
for i=13504:21254
    y3_c(i,8)=1;
end
for i=1:3430
    y3_d(i,5)=1;
end
for i=3431:9138
    y3_d(i,6)=1;
end
for i=9139:15210
    y3_d(i,7)=1;
end
for i=15211:23659
    y3_d(i,8)=1;
end
writematrix(y3_a,'experiment1.csv')
writematrix(y3_b,'experiment2.csv')
writematrix(y3_c,'experiment3.csv')
writematrix(y3_d,'experiment4.csv')