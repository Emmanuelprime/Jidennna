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

#define SPEED_CALC_INTERVAL 50000
#define PID_UPDATE_INTERVAL 10000

#define WHEEL_DIAMETER_M    0.165
#define PULSES_PER_REV      90
#define WHEEL_CIRCUMFERENCE (PI * WHEEL_DIAMETER_M)
#define METERS_PER_PULSE    (WHEEL_CIRCUMFERENCE / PULSES_PER_REV)

#define LEFT_FORWARD   HIGH
#define RIGHT_FORWARD  LOW

// PID Controller structure
struct PIDController {
  float Kp, Ki, Kd;
  float setpoint;
  float integral;
  float last_error;
  float output;
  float integral_limit;
  float output_limit;
  unsigned long last_update;
};

volatile long leftPulses  = 0;
volatile long rightPulses = 0;

volatile unsigned long lastLeftInterrupt = 0;
volatile unsigned long lastRightInterrupt = 0;

volatile bool leftDirection  = true;
volatile bool rightDirection = true;

int leftPWM = 0;
int rightPWM = 0;
float leftTargetSpeed = 0.0;
float rightTargetSpeed = 0.0;

float leftSpeed = 0.0;
float rightSpeed = 0.0;
unsigned long lastSpeedCalcTime = 0;
long lastLeftPulses = 0;
long lastRightPulses = 0;

unsigned long lastStreamTime = 0;
bool streaming = false;
bool speedControlEnabled = false;

PIDController leftPID;
PIDController rightPID;

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

void pidInit(PIDController& pid, float kp, float ki, float kd, float integral_limit, float output_limit) {
  pid.Kp = kp;
  pid.Ki = ki;
  pid.Kd = kd;
  pid.setpoint = 0;
  pid.integral = 0;
  pid.last_error = 0;
  pid.output = 0;
  pid.integral_limit = integral_limit;
  pid.output_limit = output_limit;
  pid.last_update = micros();
}

float pidUpdate(PIDController& pid, float measured_value) {
  unsigned long now = micros();
  float dt = (now - pid.last_update) / 1000000.0;
  
  if (dt < 0.001) return pid.output;
  
  pid.last_update = now;
  
  float error = pid.setpoint - measured_value;
  
  // Proportional term
  float P = pid.Kp * error;
  
  // Integral term with anti-windup
  pid.integral += error * dt;
  pid.integral = constrain(pid.integral, -pid.integral_limit, pid.integral_limit);
  float I = pid.Ki * pid.integral;
  
  // Derivative term
  float derivative = (error - pid.last_error) / dt;
  float D = pid.Kd * derivative;
  
  pid.last_error = error;
  
  // Calculate output
  pid.output = P + I + D;
  pid.output = constrain(pid.output, -pid.output_limit, pid.output_limit);
  
  return pid.output;
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
  pwm = constrain(abs(pwm), 0, MAX_PWM);
  leftPWM = pwm;
  ledcWrite(LEFT_PWM_CH, pwm);
}

void setRightPWM(int pwm) {
  pwm = constrain(abs(pwm), 0, MAX_PWM);
  rightPWM = pwm;
  ledcWrite(RIGHT_PWM_CH, pwm);
}

void updateSpeedControl() {
  if (!speedControlEnabled) return;
  
  // Update PID controllers
  float left_output = pidUpdate(leftPID, leftSpeed);
  float right_output = pidUpdate(rightPID, rightSpeed);
  
  // Apply motor feedforward based on identified models.
  // Clamp to zero below the dead-zone threshold so the motor doesn't
  // run at non-zero PWM when the setpoint is 0.
  float left_ff  = (fabs(leftPID.setpoint)  > 0.05f) ? (leftPID.setpoint  + 0.3415f) / 0.0402f : 0.0f;
  float right_ff = (fabs(rightPID.setpoint) > 0.05f) ? (rightPID.setpoint - 0.0059f) / 0.0127f : 0.0f;
  
  // Combine feedforward and PID output
  int left_pwm_cmd = constrain(round(left_ff + left_output), 0, MAX_PWM);
  int right_pwm_cmd = constrain(round(right_ff + right_output), 0, MAX_PWM);
  
  // Set motor directions based on speed sign
  if (leftPID.setpoint >= 0) {
    setLeftDirection(true);
    setLeftPWM(left_pwm_cmd);
  } else {
    setLeftDirection(false);
    setLeftPWM(left_pwm_cmd);
  }
  
  if (rightPID.setpoint >= 0) {
    setRightDirection(true);
    setRightPWM(right_pwm_cmd);
  } else {
    setRightDirection(false);
    setRightPWM(right_pwm_cmd);
  }
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

float calculateSpeed(long pulsesDelta, float deltaTimeSeconds) {
  if (deltaTimeSeconds <= 0) return 0;
  float distance = pulsesDelta * METERS_PER_PULSE;
  return distance / deltaTimeSeconds;
}

void setSpeed(float leftSpeed_ms, float rightSpeed_ms) {
  leftTargetSpeed = leftSpeed_ms;
  rightTargetSpeed = rightSpeed_ms;
  
  leftPID.setpoint = leftSpeed_ms;
  rightPID.setpoint = rightSpeed_ms;
  
  // Reset PID integral when changing setpoint
  leftPID.integral = 0;
  rightPID.integral = 0;
  
  speedControlEnabled = true;
  
  // Initialize directions based on speed sign
  setLeftDirection(leftSpeed_ms >= 0);
  setRightDirection(rightSpeed_ms >= 0);
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
    Serial.print(leftPID.setpoint, 3);
    Serial.print(",");
    Serial.print(rightPID.setpoint, 3);
    Serial.print(",");
    Serial.print(leftPID.output, 3);
    Serial.print(",");
    Serial.print(rightPID.output, 3);
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
  
  // Initialize PID controllers with tuned values
  // Left motor: K = 0.0402, tau = 0.204
  pidInit(leftPID, 10.0, 2.0, 0.1, 100, 30);
  
  // Right motor: K = 0.0127, tau = 0.219
  pidInit(rightPID, 10.0, 2.0, 0.1, 100, 30);
  
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
  
  if (speedControlEnabled) {
    updateSpeedControl();
  }
  
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
          float leftSpd = params.substring(0, commaIndex).toFloat();
          float rightSpd = params.substring(commaIndex + 1).toFloat();
          
          setSpeed(leftSpd, rightSpd);
          streaming = true;
          lastStreamTime = millis();
          
          Serial.print("STREAMING:");
          Serial.print(leftSpd, 2);
          Serial.print(",");
          Serial.println(rightSpd, 2);
        }
      }
      else if (cmd.startsWith("SPEED:")) {
        String params = cmd.substring(6);
        int commaIndex = params.indexOf(',');
        
        if (commaIndex > 0) {
          float leftSpd = params.substring(0, commaIndex).toFloat();
          float rightSpd = params.substring(commaIndex + 1).toFloat();
          
          setSpeed(leftSpd, rightSpd);
          
          Serial.print("Speed set - Left: ");
          Serial.print(leftSpd, 2);
          Serial.print(" m/s, Right: ");
          Serial.print(rightSpd, 2);
          Serial.println(" m/s");
        }
      }
      else if (cmd == "DISABLE_SPEED") {
        speedControlEnabled = false;
        setLeftPWM(0);
        setRightPWM(0);
        Serial.println("Speed control disabled");
      }
      else if (cmd.startsWith("TUNE:")) {
        String params = cmd.substring(5);
        int comma1 = params.indexOf(',');
        int comma2 = params.indexOf(',', comma1 + 1);
        
        if (comma1 > 0 && comma2 > 0) {
          float kp = params.substring(0, comma1).toFloat();
          float ki = params.substring(comma1 + 1, comma2).toFloat();
          float kd = params.substring(comma2 + 1).toFloat();
          
          pidInit(leftPID,  kp, ki, kd, 100, 30);
          pidInit(rightPID, kp, ki, kd, 100, 30);
          
          Serial.print("PID both - Kp=");
          Serial.print(kp, 2);
          Serial.print(", Ki=");
          Serial.print(ki, 2);
          Serial.print(", Kd=");
          Serial.println(kd, 2);
        }
      }
      else if (cmd.startsWith("TUNEL:")) {
        String params = cmd.substring(6);
        int comma1 = params.indexOf(',');
        int comma2 = params.indexOf(',', comma1 + 1);
        
        if (comma1 > 0 && comma2 > 0) {
          float kp = params.substring(0, comma1).toFloat();
          float ki = params.substring(comma1 + 1, comma2).toFloat();
          float kd = params.substring(comma2 + 1).toFloat();
          
          pidInit(leftPID, kp, ki, kd, 100, 30);
          
          Serial.print("PID left - Kp=");
          Serial.print(kp, 2);
          Serial.print(", Ki=");
          Serial.print(ki, 2);
          Serial.print(", Kd=");
          Serial.println(kd, 2);
        }
      }
      else if (cmd.startsWith("TUNER:")) {
        String params = cmd.substring(6);
        int comma1 = params.indexOf(',');
        int comma2 = params.indexOf(',', comma1 + 1);
        
        if (comma1 > 0 && comma2 > 0) {
          float kp = params.substring(0, comma1).toFloat();
          float ki = params.substring(comma1 + 1, comma2).toFloat();
          float kd = params.substring(comma2 + 1).toFloat();
          
          pidInit(rightPID, kp, ki, kd, 100, 30);
          
          Serial.print("PID right - Kp=");
          Serial.print(kp, 2);
          Serial.print(", Ki=");
          Serial.print(ki, 2);
          Serial.print(", Kd=");
          Serial.println(kd, 2);
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
              speedControlEnabled = false;
              setLeftDirection(true);
              setRightDirection(true);
              setLeftPWM(pwm);
              setRightPWM(pwm);
              Serial.print("Manual forward at PWM ");
              Serial.println(pwm);
            }
            break;
            
          case 'r':
          case 'R':
            if (cmd.length() > 1) {
              int pwm = cmd.substring(1).toInt();
              pwm = constrain(pwm, 0, MAX_PWM);
              speedControlEnabled = false;
              setLeftDirection(false);
              setRightDirection(false);
              setLeftPWM(pwm);
              setRightPWM(pwm);
              Serial.print("Manual reverse at PWM ");
              Serial.println(pwm);
            }
            break;
            
          case 's':
          case 'S':
            streaming = false;
            speedControlEnabled = false;
            setLeftPWM(0);
            setRightPWM(0);
            Serial.println("Motors stopped");
            break;
            
          default:
            Serial.println("Unknown command");
        }
      }
    }
  }
}