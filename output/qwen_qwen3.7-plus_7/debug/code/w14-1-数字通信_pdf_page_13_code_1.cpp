#include <SoftwareSerial.h>

// 设置软串口使用的针脚
// 7 为 RX (接蓝牙TXD), 8 为 TX (接蓝牙RXD)
SoftwareSerial softSerial(7, 8);

void setup() {
  //设定软串口波特率
  softSerial.begin(9600);
}