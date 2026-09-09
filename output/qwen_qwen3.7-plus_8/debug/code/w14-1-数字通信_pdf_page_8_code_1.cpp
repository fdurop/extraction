int val_int;      // 整形参数
float val_float;  // 浮点型参数

volatile bool updated = false;  // 参数更新标志

void setup() {
  Serial.begin(9600);
}

// 系统自动调用的串口事件中断服务程序
void serialEvent() {
  val_int = Serial.parseInt();   // 从ASCII数据流中解析整数
  val_float = Serial.parseFloat();  // 从ASCII数据流中解析浮点数

  Serial.read();  // 读取换行符'\n'

  updated = true;  // 参数已更新
}

void loop() {
  // 若参数已更新，则显示新参数
  if (updated) {
    Serial.print(val_int);
    Serial.print(" ");
    Serial.println(val_float);
    updated = false;
  }
}