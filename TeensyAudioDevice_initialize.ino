#include <Audio.h>
#include <Wire.h>
#include <SPI.h>

// 1) Create the audio objects
AudioInputAnalog      piezoIn(22);   // change parantheses depending on which pin the mic in is connceted to
AudioOutputUSB        usbOut;        // USB-Audio output

// 2) Patch the mono analog input to both USB-Audio channels
AudioConnection       patchL(piezoIn, 0, usbOut, 0);
AudioConnection       patchR(piezoIn, 0, usbOut, 1);

// 3) Allocate memory for the audio library
void setup() {
  AudioMemory(12);
  usbOut.begin();    // start USB-Audio
}

void loop() {
  // nothing needed here—audio library handles the streaming
}
