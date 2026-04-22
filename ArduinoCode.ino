#include <AFMotor.h>
AF_DCMotor motor(4);

void setup() {
  Serial.begin(9600);
	//Set initial speed of the motor & stop
	motor.setSpeed(255);
	motor.run(RELEASE);
}

void loop() {
  if(Serial.available()>0){
    char dataRX = Serial.read();
    if(dataRX=='S'){
      motor.run(FORWARD);
      Serial.println("Motor started");
    } else if(dataRX=='P'){
      motor.run(RELEASE);
      Serial.println("Motor stopped");
    }
  }
  
}
