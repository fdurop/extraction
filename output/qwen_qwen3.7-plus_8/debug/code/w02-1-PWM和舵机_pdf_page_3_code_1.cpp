void loop() {
  int cycle = 20;   // 明暗周期，单位
  float duty = 1;   // 占空比，表示点

  // digitalWrite: 向指定数字端口输出
  digitalWrite(LED_BUILTIN, HIGH);
  delay(duty * cycle);
  digitalWrite(LED_BUILTIN, LOW);
  delay(cycle * (1 - duty));
}