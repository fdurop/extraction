% 建立串口连接，端口号和波特率根据实际情况设置
s = serialport("COM4", 9600);
pause(2); % 等待串口建立

% 接收方式 1 和 2 常用于带GUI的APP
% 接收信息方式1：设置回调函数，直接给出函数语句，无需传递参数
configureCallback(s, "terminator", @(~,~) fprintf('返回信息: %s', readline(s)));

% 接收信息方式2：设置回调函数，专门定义一个函数，需传递参数
configureCallback(s, "terminator", @(~,~) receiveResponse(s));

T = 10; L = 150;
% 将T和L的值以单精度形式(4字节，对应C语言的float类型)发送
write(s, [T L], "single");
pause(1); % 等待数据发送

% 接收信息方式3：查询等待
% timeout = 1; % 设置超时时间（秒）
% startTime = tic;
% while s.NumBytesAvailable == 0 && toc(startTime) < timeout
%     pause(0.01); % 短暂暂停
% end

% if s.NumBytesAvailable > 0
%     fprintf('返回信息: %s', readline(s));
% else
%     fprintf('错误：接收超时，未收到返回信息\n');
% end

delete(s); clear s; % 串口用完后，务必释放

% 当串口有信息输入时的回调函数
function receiveResponse(s)
    fprintf('返回信息: %s', readline(s));
end