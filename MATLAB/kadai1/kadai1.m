clear
n=4096;         %�f�[�^��
dt=0.005;       %�T���v�����O�Ԋu
t=((1:n)-1)*dt;
f=t/dt/dt/n;    
fs=1/dt;        %�T���v�����O���g��
dt2=0.0025;       %�T���v�����O�Ԋu
u=((1:n)-1)*dt2;
fc_high = 10;   %�n�C�p�X�p
fc_low = 70;    %���[�p�X�p

%�Ւf���g����T���v�����Ȃǂ͏�̂��̂��g���Ă�������

%sin�g�̍쐬�ƍ����g�̍쐬
y = sin(2*pi*5*t) + sin(2*pi*50*t) + sin(2*pi*80*t);

%�t�B���^�[�̍쐬.
%�t�B���^�̎�����2�ł��܂���.
[b_l,a_l] = butter(2,fc_low/(fs/2),'low');
[b_h,a_h] = butter(2,fc_high/(fs/2),'high');
[b_p,a_p] = butter(2,[fc_high fc_low]/(fs/2),'bandpass');

% �_�E���T���v�����O�i�T���v���������炷�j
t_d = downsample(t,2);
y_d = downsample(y,2);


% �A�b�v�T���v�����O�F3���X�v���C����ԁi�T���v�����𑝂₷�j
y_u = spline(t,y,u);

% �O���t�쐬
figure(1)
subplot(2,1,1)
plot(t,y)
xlim([0,0.3])
title('3���g���̍����g�`')

subplot(2,1,2)
y1 = fft(y);
y2 = abs(y1);
plot(f,y2)
title('�����g�`��FFT')

figure(2)
subplot(2,2,1)
y_l=filter(b_l,a_l,y);
plot(t,y_l)
xlim([0,0.3])
title('���[�p�X�ʉߌ�̍����g�`')

subplot(2,2,2)
y_h=filter(b_h,a_h,y);
plot(t,y_h)
xlim([0,0.3])
title('�n�C�p�X�ʉߌ�̍����g�`')

subplot(2,2,3)
y_p=filter(b_p,a_p,y);
plot(t,y_p)
xlim([0,0.3])
title('�o���h�p�X�ʉߌ�̍����g�`')

figure(3)
subplot(2,2,1)
y1_l = fft(y_l);
y2_l = abs(y1_l);
plot(f,y2_l)
title('���[�p�X���FFT')

subplot(2,2,2)
y1_h = fft(y_h);
y2_h = abs(y1_h);
plot(f,y2_h)
title('�n�C�p�X���FFT')

subplot(2,2,3)
y1_p = fft(y_p);
y2_p = abs(y1_p);
plot(f,y2_p)
title('�o���h�p�X���FFT')

figure(4);
subplot(2,2,1)
stem(t,y,'b');
xlim([0,0.3])
title('���M��')

subplot(2,2,2)
stem(t_d,y_d,'MarkerFaceColor','r');
xlim([0,0.3])
title('�_�E���T���v�����O���ꂽ�M��')

subplot(2,2,3)
stem(t_d,y_d,'MarkerFaceColor','r');
hold on;
stem(u,y_u,'b');
xlim([0 0.3])
title('�A�b�v�T���v�����O���ꂽ�M��')

