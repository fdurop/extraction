// loop 函数仅用于实现速度控制
void loop() {
  uint32_t currentLoopTime = micros();
  if (inMoving) {
    // 循环之间的间隔
    uint32_t dt = currentLoopTime - lastLoopTime;

    // 角度累加，以脉冲数为单位
    pulse2Send += speedPerPulse * dt;

    // 是否需要步进一格。
    if (abs(pulse2Send) >= 0.5) {
      step(pulse2Send > 0); // 根据方向，步进一格

      // 已完成步进，调整累加值
      if (pulse2Send > 0) {
        pulse2Send -= 1;
      } else {
        pulse2Send += 1;
      }
    }
  }

  lastLoopTime = currentLoopTime;
}