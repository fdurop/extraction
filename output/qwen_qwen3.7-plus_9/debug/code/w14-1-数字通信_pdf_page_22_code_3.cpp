void loop() {
  if (flagRecv) {
    flagRecv = 0;  // clear flag
    while (CAN_MSGAVAIL == CAN.checkReceive()) {
      // 读取数据到缓冲区
      CAN.readMsgBuf(&len, recvBuf);
      // 把缓冲区二进制数据转换为浮点数
      float pos;
      memcpy(&pos, &recvBuf[0], len);
      Serial.println(pos);
      // 返回处理成功的信息
      CAN.MCP_CAN::sendMsgBuf(slaveID, 0, 2, "OK");
    }
  }
}