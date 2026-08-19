#include <TimerOne.h>

void setup() {
  //初始化LED端口
  pinMode(LED_BUILTIN, OUTPUT);

  // 每 1 s 触发一次时间中断
  Timer1.initialize(long(1 * 1E6));

  // 设置中断响应函数
  Timer1.attachInterrupt(LED);

  // 启动时间中断
  Timer1.start();
}

void LED() {
  static bool light = true;

  digitalWrite(LED_BUILTIN, light);
  light = !light;
}