float inputPos() {
  while (!Serial.available())
    ;

  // 从ASCII数据流中解析浮点数
  float res = Serial.parseFloat();
  // 读取换行符 '\n'
  Serial.read();
  return res;
}