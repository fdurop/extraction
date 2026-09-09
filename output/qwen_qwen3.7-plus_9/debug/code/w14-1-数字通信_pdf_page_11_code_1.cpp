volatile bool dataReady = false;
float T;
float L;

void setup(){
  Serial.begin(9600);
  while (!Serial) {}  // 等待串口初始化完成

  pinMode(LED_BUILTIN, OUTPUT);
}

// 方式1: 使用串口中断来接收数据，并非所有开发板都支持
void serialEvent() {
  if (Serial.available()) {
    Serial.readBytes((byte *)&T, sizeof(float));
    Serial.readBytes((byte *)&L, sizeof(float));
    dataReady = true;
  }
}

void loop() {
  // ****************************************************
  // 方式2: 查询等待，通用性好，效率低
  while (!Serial.available())
    ;

  Serial.readBytes((byte *)&T, sizeof(float));
  Serial.readBytes((byte *)&L, sizeof(float));
  dataReady = true;
  // ****************************************************/
  if (dataReady) {
    // 发送确认信息
    Serial.print("Uno received: ");
    Serial.print(T, 6);  // 显示6位小数
    Serial.print(", ");
    Serial.println(L, 6);  // 显示6位小数
    dataReady = false;
  }
}