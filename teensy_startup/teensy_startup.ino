const int inputPin = 22; // Analog input pin
const int bufferSize = 44100; // Size of the buffer
const int batchSize = 100; // Number of samples to send in one batch
int audioBuffer[bufferSize]; // Audio buffer
int bufferIndex = 0; // Current index for filling the buffer

void setup() {
  Serial.begin(115200); // Initialize serial communication at 115200 baud rate
}

void loop() {
  // Read and store the value in the buffer
  if (bufferIndex < bufferSize) {
    audioBuffer[bufferIndex++] = analogRead(inputPin); // Read value from analog pin
  } else {
    // Buffer is full; send data in batches
    sendBufferData();
    bufferIndex = 0; // Reset buffer index for next batch
  }
  
  // Check if we have enough data to send a batch
  if (bufferIndex >= batchSize) {
    sendBatchData();
  }

  delayMicroseconds(4); // Maintain ~44.1 kHz sample rate (22 microseconds per sample)
}

// Function to send data in batches
void sendBatchData() {
  for (int i = 0; i < batchSize; i++) {
    Serial.println(audioBuffer[i]);
  }
  // Shift remaining data to the front of the buffer
  for (int i = batchSize; i < bufferIndex; i++) {
    audioBuffer[i - batchSize] = audioBuffer[i];
  }
  bufferIndex -= batchSize; // Adjust buffer index after sending
}

// Function to send the entire buffer data when full
void sendBufferData() {
  for (int i = 0; i < bufferSize; i++) {
    Serial.println(audioBuffer[i]);
  }
}