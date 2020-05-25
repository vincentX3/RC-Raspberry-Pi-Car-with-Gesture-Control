
#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#define GO_AHEAD 3 
#define BACK_UP 2
#define TURN_LEFT 0
#define TURN_RIGHT 1
#define ALL_IDLE 4

// personal WIFI SSID and password config.
#ifndef STASSID
#define STASSID "FANG"
#define STAPSK  "FANG7318W"


#endif
// static IP setting.
//IPAddress local_ip(192,168,43,200);
//IPAddress gateway(192,168,43,1);
//IPAddress subnet(255,255,255,0);

unsigned int localPort = 8888;      // local port to listen on

// buffers for receiving and sending data
char packetBuffer[UDP_TX_PACKET_MAX_SIZE]; //buffer to hold incoming packet,
char  ReplyBuffer[] = "test\r\n";       // a string to send back

//port for control
int en_port = 4;
int left_1 = 13;
int left_2 = 12;
int right_1 = 14;
int right_2 = 16;

WiFiUDP Udp;

void PInSet()
{
  pinMode(left_1,OUTPUT);
  pinMode(left_2,OUTPUT);
  pinMode(right_1,OUTPUT);
  pinMode(right_2,OUTPUT);
  pinMode(en_port,OUTPUT);
  digitalWrite(en_port,HIGH);
  digitalWrite(left_1,LOW); 
  digitalWrite(left_2,LOW);
  digitalWrite(right_1,LOW);
  digitalWrite(right_2,LOW);
}

void setup() {
  Serial.begin(115200);
//  WiFi.config(local_ip,gateway,subnet); // static ip setting
  WiFi.mode(WIFI_STA);
  WiFi.begin(STASSID, STAPSK);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(500);
  }
  Serial.println(WiFi.localIP());
  Udp.begin(localPort);
  Serial.print("start to listen.\n");
  PInSet();
  Serial.print("set pinmode.\n"); 
}

void Operate(int i)
{
  switch(i){
    case GO_AHEAD:
      digitalWrite(left_1,HIGH); 
      digitalWrite(left_2,LOW);
      digitalWrite(right_1,HIGH);
      digitalWrite(right_2,LOW);
      delay(200);
      digitalWrite(left_1,LOW); 
      digitalWrite(left_2,LOW);
      digitalWrite(right_1,LOW);
      digitalWrite(right_2,LOW);
      break;
    case BACK_UP:
      digitalWrite(left_1,LOW); 
      digitalWrite(left_2,HIGH);
      digitalWrite(right_1,LOW);
      digitalWrite(right_2,HIGH);
      delay(200);
      digitalWrite(left_1,LOW); 
      digitalWrite(left_2,LOW);
      digitalWrite(right_1,LOW);
      digitalWrite(right_2,LOW);
      break;
    case TURN_LEFT:
      digitalWrite(left_1,LOW); 
      digitalWrite(left_2,HIGH);
      digitalWrite(right_1,HIGH);
      digitalWrite(right_2,LOW);
      delay(200);
      digitalWrite(left_1,LOW); 
      digitalWrite(left_2,LOW);
      digitalWrite(right_1,LOW);
      digitalWrite(right_2,LOW);
      break;
    case TURN_RIGHT:
      digitalWrite(left_1,HIGH); 
      digitalWrite(left_2,LOW);
      digitalWrite(right_1,LOW);
      digitalWrite(right_2,HIGH);
      delay(200);
      digitalWrite(left_1,LOW); 
      digitalWrite(left_2,LOW);
      digitalWrite(right_1,LOW);
      digitalWrite(right_2,LOW);
      break;
    case ALL_IDLE:
      digitalWrite(left_1,LOW); 
      digitalWrite(left_2,LOW);
      digitalWrite(right_1,LOW);
      digitalWrite(right_2,LOW);
      break;    
  }
}

void loop() {
  // if there's data available, read a packet
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    int len = Udp.read(packetBuffer,UDP_TX_PACKET_MAX_SIZE);
    if(len > 0){
      Serial.print(packetBuffer);
      // send a reply, to the IP address and port that sent us the packet we received
      if(packetBuffer[0]=='t'){
      Udp.beginPacket(Udp.remoteIP(), Udp.remotePort());
      Udp.write(ReplyBuffer);
      Udp.endPacket();
      delay(10);
      }

      //go_ahead
      if(packetBuffer[0]=='3'){
        Serial.print("go ahead.\n");
        Operate(GO_AHEAD);
        }

      //back_up
      if(packetBuffer[0]=='2'){
        Serial.print("back.\n");
        Operate(BACK_UP);
        }

      //turn _left
      if(packetBuffer[0]=='0'){
        Serial.print("turn left.\n");
        Operate(TURN_LEFT);
        }

      //turn _right
      if(packetBuffer[0]=='1'){
        Serial.print("turn right.\n");
        Operate(TURN_RIGHT);
        }

      //idle
      if(packetBuffer[0]=='4'){
        Serial.print("idle.\n");
        Operate(ALL_IDLE);
        }
    }
  }
}
