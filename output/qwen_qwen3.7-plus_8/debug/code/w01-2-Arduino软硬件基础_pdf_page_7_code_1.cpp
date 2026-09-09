// the setup function runs once when you press reset or power the board
void setup() {
  Serial.begin(9600);   // 初始化串口，默认波特率为 9600
  while (!Serial) {};   // 等待串口初始化完成

  Serial.println("Hello Arduino!"); // Serial.print("Hello Arduino!\n")
}

void loop() {
  // put your main code here, to run repeatedly:
}