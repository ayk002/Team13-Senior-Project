const int inputPin = 23; // Analog input pin
const int bufferSize = 44100; // Adjust size as needed
int audioBuffer[bufferSize];
int bufferIndex = 0;


void setup() {
 Serial.begin(115200); // initialize serial communication at 115200 baud rate
}


void loop() {
 if (bufferIndex < bufferSize) {
   audioBuffer[bufferIndex++] = analogRead(inputPin); // Read and store the value
 } else {
   // When the buffer is full, send the data over Serial
   for (int i = 0; i < bufferSize; i++) {
     Serial.println(audioBuffer[i]);
   }
   bufferIndex = 0; // Reset buffer index
 }
 delayMicroseconds(22); // Sample rate ~44.1 kHz (22 microseconds per sample)
}
