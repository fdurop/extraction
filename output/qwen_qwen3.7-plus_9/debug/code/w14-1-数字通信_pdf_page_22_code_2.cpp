#include <SPI.h>
#include "mcp2515_can.h"

const int SPI_CS_PIN = 9;
const int CAN_INT_PIN = 2;
mcp2515_can CAN(SPI_CS_PIN);  // Set CS pin

unsigned char recvBuf[8];  // 接受信息缓存
uint8_t len;               // 接收到的字节数
const uint8_t slaveID = 1; // 下位机 CAN ID
bool flagRecv;

void setup() {
  Serial.begin(1000000);
  while (!Serial) {};

  // Init can bus : baudrate = 1000k
  while (CAN_OK != CAN.begin(CAN_1000KBPS)) {
    Serial.println("CAN init fail, retry...");
    delay(100);
  }
  Serial.println("CAN init ok!\n");
  // 设置监听CAN的外部中断
  attachInterrupt(digitalPinToInterrupt(CAN_INT_PIN), posMonitor, FALLING);
}

// 监听CAN的外部中断
void posMonitor() {
  flagRecv = true;
}