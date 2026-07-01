#include <Arduino.h>

#define LEFT_PWM     25
#define LEFT_DIR     26
#define LEFT_SC      34

#define RIGHT_PWM    27
#define RIGHT_DIR    33
#define RIGHT_SC     35

#define LEFT_PWM_CH   2
#define RIGHT_PWM_CH  1
#define PWM_FREQ      1000
#define PWM_RES       8
#define MAX_PWM       60

#define DIR_CHANGE_DELAY   300
#define RAMP_STEP_DELAY    10
#define RAMP_STEP_SIZE     5
#define DEBOUNCE_TIME_US   1000

#define SPEED_CALC_INTERVAL 250000

#define WHEEL_DIAMETER_M    0.165
#define PULSES_PER_REV      90
#define WHEEL_CIRCUMFERENCE (PI * WHEEL_DIAMETER_M)
#define METERS_PER_PULSE    (WHEEL_CIRCUMFERENCE / PULSES_PER_REV)

#define LEFT_FORWARD   HIGH
#define RIGHT_FORWARD  LOW

volatile long leftPulses  = 0;
volatile long rightPulses = 0;

volatile unsigned long lastLeftInterrupt = 0;
volatile unsigned long lastRightInterrupt = 0;

volatile bool leftDirection  = true;
volatile bool rightDirection = true;

int leftPWM = 0;
int rightPWM = 0;
int targetLeftPWM = 0;
int targetRightPWM = 0;
bool targetLeftDir = true;
bool targetRightDir = true;

bool decelerating = false;
bool accelerating = false;
unsigned long stateTimer = 0;

float leftSpeed = 0.0;
float rightSpeed = 0.0;
unsigned long lastSpeedCalcTime = 0;
long lastLeftPulses = 0;
long lastRightPulses = 0;

unsigned long lastPrintTime = 0;
unsigned long lastStreamTime = 0;
bool streaming = false;

void IRAM_ATTR leftISR() { 
  unsigned long now = micros();
  if (now - lastLeftInterrupt >= DEBOUNCE_TIME_US) {
    lastLeftInterrupt = now;
    if (leftDirection) {
      leftPulses++;
    } else {
      leftPulses--;
    }
  }
}

void IRAM_ATTR rightISR() { 
  unsigned long now = micros();
  if (now - lastRightInterrupt >= DEBOUNCE_TIME_US) {
    lastRightInterrupt = now;
    if (rightDirection) {
      rightPulses++;
    } else {
      rightPulses--;
    }
  }
}

void setLeftDirection(bool forward) {
  digitalWrite(LEFT_DIR, forward ? LEFT_FORWARD : !LEFT_FORWARD);
  leftDirection = forward;
}

void setRightDirection(bool forward) {
  digitalWrite(RIGHT_DIR, forward ? RIGHT_FORWARD : !RIGHT_FORWARD);
  rightDirection = forward;
}

void setLeftPWM(int pwm) {
  pwm = constrain(pwm, 0, MAX_PWM);
  leftPWM = pwm;
  ledcWrite(LEFT_PWM_CH, pwm);
}

void setRightPWM(int pwm) {
  pwm = constrain(pwm, 0, MAX_PWM);
  rightPWM = pwm;
  ledcWrite(RIGHT_PWM_CH, pwm);
}

void setMotors(int leftPwm, int rightPwm) {
  targetLeftDir = (leftPwm >= 0);
  targetRightDir = (rightPwm >= 0);
  targetLeftPWM = constrain(abs(leftPwm), 0, MAX_PWM);
  targetRightPWM = constrain(abs(rightPwm), 0, MAX_PWM);
  
  if (targetLeftPWM == 0 && targetRightPWM == 0) {
    if (leftPWM > 0 || rightPWM > 0) {
      decelerating = true;
      accelerating = false;
      stateTimer = millis();
    }
    return;
  }
  
  if (leftPWM > 0 || rightPWM > 0) {
    if (targetLeftDir != leftDirection || targetRightDir != rightDirection) {
      decelerating = true;
      accelerating = false;
      stateTimer = millis();
      return;
    }
  } else {
    if (targetLeftDir != leftDirection || targetRightDir != rightDirection) {
      setLeftDirection(targetLeftDir);
      setRightDirection(targetRightDir);
      accelerating = true;
      decelerating = false;
      stateTimer = millis();
      return;
    }
  }
  
  decelerating = false;
  accelerating = false;
}

void updateSpeeds() {
  unsigned long currentTime = micros();
  
  if (currentTime - lastSpeedCalcTime >= SPEED_CALC_INTERVAL) {
    noInterrupts();
    long currentLeftPulses = leftPulses;
    long currentRightPulses = rightPulses;
    interrupts();
    
    float dt = (currentTime - lastSpeedCalcTime) / 1000000.0;
    
    leftSpeed = calculateSpeed(currentLeftPulses - lastLeftPulses, dt);
    rightSpeed = calculateSpeed(currentRightPulses - lastRightPulses, dt);
    
    lastLeftPulses = currentLeftPulses;
    lastRightPulses = currentRightPulses;
    lastSpeedCalcTime = currentTime;
  }
}

void updateRamp() {
  unsigned long now = millis();
  
  if (decelerating) {
    if (leftPWM > 0 || rightPWM > 0) {
      if (leftPWM > 0) setLeftPWM(max(leftPWM - RAMP_STEP_SIZE, 0));
      if (rightPWM > 0) setRightPWM(max(rightPWM - RAMP_STEP_SIZE, 0));
      delayWithSpeedUpdate(RAMP_STEP_DELAY);
    } else {
      decelerating = false;
      if (targetLeftPWM > 0 || targetRightPWM > 0) {
        setLeftDirection(targetLeftDir);
        setRightDirection(targetRightDir);
        accelerating = true;
        stateTimer = millis();
      }
    }
  }
  else if (accelerating) {
    bool leftDone = (leftPWM >= targetLeftPWM);
    bool rightDone = (rightPWM >= targetRightPWM);
    
    if (!leftDone || !rightDone) {
      if (!leftDone) setLeftPWM(min(leftPWM + RAMP_STEP_SIZE, targetLeftPWM));
      if (!rightDone) setRightPWM(min(rightPWM + RAMP_STEP_SIZE, targetRightPWM));
      delayWithSpeedUpdate(RAMP_STEP_DELAY);
    } else {
      accelerating = false;
    }
  }
  else {
    bool leftChanged = (leftPWM != targetLeftPWM);
    bool rightChanged = (rightPWM != targetRightPWM);
    
    if (leftChanged || rightChanged) {
      if (leftPWM < targetLeftPWM) {
        setLeftPWM(min(leftPWM + RAMP_STEP_SIZE, targetLeftPWM));
      } else if (leftPWM > targetLeftPWM) {
        setLeftPWM(max(leftPWM - RAMP_STEP_SIZE, targetLeftPWM));
      }
      if (rightPWM < targetRightPWM) {
        setRightPWM(min(rightPWM + RAMP_STEP_SIZE, targetRightPWM));
      } else if (rightPWM > targetRightPWM) {
        setRightPWM(max(rightPWM - RAMP_STEP_SIZE, targetRightPWM));
      }
      delayWithSpeedUpdate(RAMP_STEP_DELAY);
    }
  }
}

void delayWithSpeedUpdate(int ms) {
  unsigned long start = millis();
  while (millis() - start < ms) {
    updateSpeeds();
    delay(1);
  }
}

float calculateSpeed(long pulsesDelta, float deltaTimeSeconds) {
  if (deltaTimeSeconds <= 0) return 0;
  float distance = pulsesDelta * METERS_PER_PULSE;
  return distance / deltaTimeSeconds;
}

void streamData() {
  unsigned long currentTime = millis();
  
  if (streaming && (currentTime - lastStreamTime >= 100)) {
    lastStreamTime = currentTime;
    
    noInterrupts();
    long lCount = leftPulses;
    long rCount = rightPulses;
    interrupts();
    
    Serial.print("DATA,");
    Serial.print(currentTime);
    Serial.print(",");
    Serial.print(lCount);
    Serial.print(",");
    Serial.print(rCount);
    Serial.print(",");
    Serial.print(leftSpeed, 3);
    Serial.print(",");
    Serial.print(rightSpeed, 3);
    Serial.print(",");
    Serial.print(leftPWM);
    Serial.print(",");
    Serial.print(rightPWM);
    Serial.print(",");
    Serial.print(leftDirection ? "1" : "-1");
    Serial.print(",");
    Serial.print(rightDirection ? "1" : "-1");
    Serial.println();
  }
}

void setup() {
  Serial.begin(115200);
  
  pinMode(LEFT_DIR, OUTPUT);
  pinMode(RIGHT_DIR, OUTPUT);
  
  pinMode(LEFT_SC, INPUT_PULLUP);
  pinMode(RIGHT_SC, INPUT_PULLUP);
  
  attachInterrupt(digitalPinToInterrupt(LEFT_SC), leftISR, RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_SC), rightISR, RISING);
  
  ledcSetup(LEFT_PWM_CH, PWM_FREQ, PWM_RES);
  ledcSetup(RIGHT_PWM_CH, PWM_FREQ, PWM_RES);
  ledcAttachPin(LEFT_PWM, LEFT_PWM_CH);
  ledcAttachPin(RIGHT_PWM, RIGHT_PWM_CH);
  
  setLeftDirection(true);
  setRightDirection(true);
  setLeftPWM(0);
  setRightPWM(0);
  
  lastSpeedCalcTime = micros();
  lastLeftPulses = 0;
  lastRightPulses = 0;
  
  Serial.println("READY");
}

void loop() {
  updateSpeeds();
  updateRamp();
  streamData();
  
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.length() > 0) {
      if (cmd == "PING") {
        Serial.println("READY");
      }
      else if (cmd == "STOP_STREAM") {
        streaming = false;
        Serial.println("Stream stopped");
      }
      else if (cmd.startsWith("STREAM:")) {
        String params = cmd.substring(7);
        int commaIndex = params.indexOf(',');
        
        if (commaIndex > 0) {
          int leftPwm = params.substring(0, commaIndex).toInt();
          int rightPwm = params.substring(commaIndex + 1).toInt();
          
          leftPwm = constrain(leftPwm, -MAX_PWM, MAX_PWM);
          rightPwm = constrain(rightPwm, -MAX_PWM, MAX_PWM);
          
          setMotors(leftPwm, rightPwm);
          streaming = true;
          lastStreamTime = millis();
          
          Serial.print("STREAMING:");
          Serial.print(leftPwm);
          Serial.print(",");
          Serial.println(rightPwm);
        }
      }
      else {
        char command = cmd.charAt(0);
        
        switch (command) {
          case 'f':
          case 'F':
            if (cmd.length() > 1) {
              int pwm = cmd.substring(1).toInt();
              pwm = constrain(pwm, 0, MAX_PWM);
              setMotors(pwm, pwm);
              Serial.print("Moving forward at PWM ");
              Serial.println(pwm);
            }
            break;
            
          case 'r':
          case 'R':
            if (cmd.length() > 1) {
              int pwm = cmd.substring(1).toInt();
              pwm = constrain(pwm, 0, MAX_PWM);
              setMotors(-pwm, -pwm);
              Serial.print("Moving reverse at PWM ");
              Serial.println(pwm);
            }
            break;
            
          case 's':
          case 'S':
            streaming = false;
            setMotors(0, 0);
            Serial.println("Stopping motors");
            break;
            
          default:
            Serial.println("Unknown command");
        }
      }
    }
  }
}