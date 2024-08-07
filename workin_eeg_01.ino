#include <Servo.h>
#define CH1 8 
#define CH2 12 
#define CH3 10 
#define CH4 11
Servo servo_6;


void setup(){
  Serial.begin(9600);

  pinMode(CH1, OUTPUT);
  pinMode(CH2, OUTPUT);
  pinMode(CH3, OUTPUT);
  pinMode(CH4, OUTPUT);
  
  digitalWrite(CH1,HIGH);
  digitalWrite(CH2,HIGH);
  digitalWrite(CH3,HIGH);
  digitalWrite(CH4,HIGH);

  servo_6.attach(9);
  


}

void loop (){
while(Serial.available()){
  
  char ser = Serial.read();
  if (ser == '1'){
    if(digitalRead(CH1)== LOW){offPin(CH1);}
    else{onPin(CH1);}
  }
  else if (ser == '2'){
    if(digitalRead(CH2)== LOW){offPin(CH2);}
    else{onPin(CH2);}
  }
  else if (ser == '3'){
    if(digitalRead(CH3)== LOW){offPin(CH3);}
    else{onPin(CH3);}
  }
  else if (ser == '4'){
    if(digitalRead(CH4)== LOW){offPin(CH4);}
    else{onPin(CH4);}
  
  }
  else if(ser == '5'){
    servo_6.write(60);
  }
  else if(ser == '6'){
    servo_6.write(0);
  }


}




}



void onPin(int8_t pin) {
  digitalWrite(pin, LOW);
}

void offPin(int8_t pin) {
  digitalWrite(pin, HIGH);
}
