// the setup function runs once when you press reset or power the board
void setup() {
  // pinMode: 初始化数字端口
  pinMode(LED_BUILTIN, OUTPUT);   // 使用Uno时，LED_BUILTIN 为 port 13.
}

// the loop function runs over and over again forever
void loop() {
  int cycle = 20;    // 明暗周期，单位 ms. 可尝试较小数值，如 20 ms
  float duty = 1;    // 占空比，表示点亮时长与cycle的比值，在0-1之间变化

  // digitalWrite: 向指定数字端口输出高/低电平
  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(duty * cycle);               // wait for a second, unit: ms
  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
  delay(cycle * (1 - duty));         // wait for a second, unit: ms
}