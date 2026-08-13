机电系统原理及实践—— 直线运动平台（上）
机电智控实验室tangguoan@fudan.edu.cn mmguo@fudan.edu.cn
§2.1 限位开关
又称行程开关，对运动机构进行限制，使其停止或转向。
有触点，机械式接触。
主要用于控制回路，通过控制电路实现对机械设备的限位保护
和运动控制。
实践：限位开关的使用。
DI：Digital Input
用digitalRead(端口号)检测限位开关。
用Serial.print()显示开关状态。
12-5
COM
NO
NC
限位时，NO与COM导通
GND
DI
DI
GND
D11
UNO
COM
NO
NC
限位开关1
COM
NO
NC
限位开关2
D12
GND
GND