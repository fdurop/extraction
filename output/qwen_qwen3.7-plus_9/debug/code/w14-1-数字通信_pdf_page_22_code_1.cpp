#include <SPI.h>
#include "mcp2515_can.h"

// For Arduino MCP2515 Hat:
// the cs pin of the version after v1.1 is default to D9
// v0.9b and v1.0 is default D10
const int SPI_CS_PIN = 9;
const int CAN_INT_PIN = 2;
mcp2515_can CAN(SPI_CS_PIN);  // Set CS pin

unsigned char recvBuf[8];    // 接受信息缓存
uint8_t len;                 // 接收到的字节数
const uint8_t slaveID = 1;   // 下位机 CAN ID

void setup() {
  Serial.begin(1000000);
  while (!Serial) {};
  // init can bus : baudrate = 1000k
  while (CAN_OK != CAN.begin(CAN_1000KBPS)) {
    Serial.println("CAN init fail, retry...");
    delay(100);
  }
  Serial.println("CAN init ok!\n");

  // 从串口输入一个位置
  float pos = inputPos();
  // 发送 pos 给下位机
  CAN_MCP_CAN::sendMsgBuf(slaveID, 0, sizeof(pos), (byte *)&pos);
  // 等待下位机返回信息
  while (CAN_MSGAVAIL != CAN.checkReceive())
    ;
  // 读取下位机返回的信息，并在串口输出
  len = 0;
  CAN.readMsgBuf(&len, recvBuf);
  int id = CAN.getCanId();
  Serial.print("id : "); Serial.println(id);
  Serial.println((char *)recvBuf);
}