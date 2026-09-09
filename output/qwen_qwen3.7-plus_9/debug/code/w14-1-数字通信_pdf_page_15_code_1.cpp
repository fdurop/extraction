#include <Wire.h>

// 下位机 I2C 地址
const byte slaveAdress = 20;

void setup() {
  Serial.begin(9600);
  while (!Serial) {}

  float ang = PI;

  // 作为主机启动 I2C 通信
  Wire.begin();
  // 开始向下位机发送信息
  Wire.beginTransmission(slaveAdress);
  // 发送具体数据
  Wire.write((byte*)&ang, sizeof(ang));
  // 结束发送
  Wire.endTransmission();

  // 向下位机请求 1 字节数据
  Wire.requestFrom(slaveAdress, 1);
  // 等待反馈
  while (!Wire.available())
    ;
  // 读取下位机发送的 1 个字节
  bool res = Wire.read();

  Serial.println(res);
}