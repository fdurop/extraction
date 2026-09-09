#include <Wire.h>

// 下位机 I2C 地址
const byte slaveAdress = 20;
bool res;

void setup() {
  Serial.begin(9600);
  while (!Serial) {};

  // 作为下位机启动 I2C 通信
  Wire.begin(slaveAdress);
  // 设置接收数据的中断处理函数
  Wire.onReceive(receiveEvent);
  // 设置发送数据的中断处理函数
  Wire.onRequest(requestEvent);
}

// 接收数据的中断处理函数
void receiveEvent(int howMany) {
  float ang;
  // 读取上位机发来的数据
  Wire.readBytes((byte *)&ang, sizeof(float));
  Serial.println(ang);
  if (abs(ang - PI) < 1E-6)
    res = true;
  else
    res = false;
}

// 发送数据的中断处理函数
void requestEvent() {
  Wire.write(res);
}