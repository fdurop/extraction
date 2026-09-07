Buttons.NumButtons = 3;
ButtonMaping(0, pinSHP, SliderSpeed_SHP);
ButtonMaping(1, pinVBR, SliderSpeed_VBR);
ButtonMaping(2, pinIPL, SliderSpeed_IPL);
for ( int i = 0; i < Buttons.NumButtons; i++ ) {
pinMode(Buttons.pinBUT[i], INPUT_PULLUP);
}
Buttons.RUNING_MODE = Buttons.NONE;
void ButtonMaping(int sn, int pin, float (*cbf)(float)) {
}