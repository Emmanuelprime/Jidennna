#include <MPU6050_tockn.h>
#include <Wire.h>

#define LEFT_PWM      9
#define LEFT_DIR      8
#define LEFT_SC       2

#define RIGHT_PWM     10
#define RIGHT_DIR     7
#define RIGHT_SC      3

#define MAX_PWM        60
#define LEFT_FORWARD   HIGH
#define RIGHT_FORWARD  LOW
#define WHEEL_DIAMETER_M  0.165f
#define PULSES_PER_REV    45
#define METERS_PER_PULSE  (PI * WHEEL_DIAMETER_M / PULSES_PER_REV)

// Robot dimensions
#define WHEELBASE_M  0.52f

// ─── MPU6050 FILTER AND DEADZONE SETTINGS ──────────────────────────────────

#define HEADING_DEADZONE_DEG  0.3f
#define TURN_RATE_DEADZONE    0.015f
#define YAW_FILTER_ALPHA  0.92f
#define RATE_FILTER_ALPHA  0.5f

// MPU6050 object
MPU6050 mpu6050(Wire);

// ─── PID GAINS ──────────────────────────────────────────────────────────────

#define KP_HEADING 1.2
#define KI_HEADING 0.01
#define KD_HEADING 0.08

#define KP_TURN_RATE 1.8
#define KI_TURN_RATE 0.03
#define KD_TURN_RATE 0.08

#define KP_LEFT  1.8   
#define KI_LEFT  0.02  
#define KD_LEFT  0.15  

#define KP_RIGHT 2.5   
#define KI_RIGHT 0.03  
#define KD_RIGHT 0.20  

// ─── MOTOR CHARACTERIZATION ──────────────────────────────────────────────────

#define LEFT_FWD_SLOPE     0.0209f
#define LEFT_FWD_INTERCEPT -0.0167f
#define LEFT_REV_SLOPE     0.0209f
#define LEFT_REV_INTERCEPT -0.0167f

#define RIGHT_FWD_SLOPE    0.0209f
#define RIGHT_FWD_INTERCEPT -0.0167f
#define RIGHT_REV_SLOPE    0.0209f
#define RIGHT_REV_INTERCEPT -0.0167f

#define RIGHT_FWD_COMP  1.30f
#define LEFT_REV_COMP   1.30f

#define LEFT_DEADZONE 10
#define RIGHT_DEADZONE 10

#define CONTROL_INTERVAL 50
#define SAMPLE_TIME 0.05

volatile unsigned long debounceUS = 1000UL;

// ─── ENCODER VARIABLES ──────────────────────────────────────────────────────

volatile long          leftPulses  = 0;
volatile unsigned long lastLeftUS  = 0;
volatile bool          leftFwd     = true;

volatile long          rightPulses  = 0;
volatile unsigned long lastRightUS  = 0;
volatile bool          rightFwd     = true;

int leftCurrentPWM = 0;
int rightCurrentPWM = 0;

float targetLeftSpeed = 0.0;
float targetRightSpeed = 0.0;
float currentLeftSpeed = 0.0;
float currentRightSpeed = 0.0;

float leftIntegral = 0;
float rightIntegral = 0;
float leftPrevError = 0;
float rightPrevError = 0;

float filteredLeftSpeed = 0;
float filteredRightSpeed = 0;

unsigned long lastControlTime = 0;

float targetLinearVelocity = 0.0;
float targetAngularVelocity = 0.0;

float leftFeedforward = 0;
float rightFeedforward = 0;
float leftPidCorrection = 0;
float rightPidCorrection = 0;

// ─── HEADING CONTROL ──────────────────────────────────────────────────────

float headingIntegral = 0;
float headingPrevError = 0;
float targetYaw = 0;
bool headingInitialized = false;
float currentYaw = 0;
float filteredYaw = 0;

// ─── TURN RATE CONTROL ──────────────────────────────────────────────────────

float turnRateIntegral = 0;
float turnRatePrevError = 0;
float actualAngularVelocity = 0;
float filteredAngularVelocity = 0;
float previousYaw = 0;
unsigned long lastYawTime = 0;
bool turnRateControlActive = false;

unsigned long lastIMUUpdate = 0;

// ─── ODOMETRY ──────────────────────────────────────────────────────────────

// Position in meters
float odomX = 0.0;
float odomY = 0.0;
float odomYaw = 0.0;

// Previous odometry values for delta calculation
float prevOdomX = 0.0;
float prevOdomY = 0.0;
float prevOdomYaw = 0.0;

// Distance traveled
float totalDistance = 0.0;

// ─── SOFT DEADZONE ──────────────────────────────────────────────────────────

float softDeadzone(float error, float threshold) {
  if (abs(error) < threshold) {
    return error * (abs(error) / threshold);
  }
  return error;
}

// ─── EXPONENTIAL MOVING AVERAGE ────────────────────────────────────────────

float expMovingAverage(float newValue, float prevValue, float alpha) {
  return alpha * newValue + (1.0f - alpha) * prevValue;
}

// ─── ODOMETRY UPDATE ────────────────────────────────────────────────────────

void updateOdometry(float dt) {
  // Calculate robot velocity in robot frame
  float v = (currentLeftSpeed + currentRightSpeed) / 2.0f;
  float w = (currentRightSpeed - currentLeftSpeed) / WHEELBASE_M;
  
  // Use IMU for heading when available, otherwise use wheel odometry
  if (headingInitialized) {
    odomYaw = filteredYaw * (PI / 180.0);  // Convert degrees to radians
  } else {
    // Fallback to wheel odometry heading
    static float wheelYaw = 0;
    wheelYaw += w * dt;
    odomYaw = wheelYaw;
  }
  
  // Update position using kinematic equations
  // Using the average of current and previous velocity for better accuracy
  float deltaX = v * cos(odomYaw) * dt;
  float deltaY = v * sin(odomYaw) * dt;
  
  odomX += deltaX;
  odomY += deltaY;
  
  // Update total distance
  totalDistance += abs(v) * dt;
  
  // Store previous values for delta calculation
  prevOdomX = odomX;
  prevOdomY = odomY;
  prevOdomYaw = odomYaw;
}

// ─── MPU6050 Functions ──────────────────────────────────────────────────────

void initMPU6050() {
  Wire.begin();
  mpu6050.begin();
  mpu6050.calcGyroOffsets(true);
  
  Serial.println("# MPU6050 Initialized and Calibrated!");
  
  mpu6050.update();
  currentYaw = mpu6050.getAngleZ();
  filteredYaw = currentYaw;
  targetYaw = currentYaw;
  previousYaw = currentYaw;
  lastYawTime = micros();
  headingInitialized = true;
  
  // Initialize odometry
  odomX = 0.0;
  odomY = 0.0;
  odomYaw = filteredYaw * (PI / 180.0);
}

void updateIMU() {
  mpu6050.update();
  currentYaw = mpu6050.getAngleZ();
  
  unsigned long now = micros();
  float dt = (now - lastYawTime) / 1000000.0f;
  if (dt > 0.001 && dt < 0.1) {
    float yawDelta = currentYaw - previousYaw;
    while (yawDelta > 180) yawDelta -= 360;
    while (yawDelta < -180) yawDelta += 360;
    float rawRate = yawDelta / dt * (PI / 180.0);
    
    filteredAngularVelocity = expMovingAverage(rawRate, filteredAngularVelocity, RATE_FILTER_ALPHA);
    filteredYaw = expMovingAverage(currentYaw, filteredYaw, YAW_FILTER_ALPHA);
  }
  previousYaw = currentYaw;
  lastYawTime = now;
}

// ─── Heading PID ────────────────────────────────────────────────────────────

float computeHeadingPID(float target, float current, float dt) {
  float error = target - current;
  
  while (error > 180) error -= 360;
  while (error < -180) error += 360;
  
  error = softDeadzone(error, HEADING_DEADZONE_DEG);
  
  headingIntegral += error * dt;
  headingIntegral = constrain(headingIntegral, -5.0, 5.0);
  
  float derivative = (error - headingPrevError) / dt;
  float output = KP_HEADING * error + KI_HEADING * headingIntegral + KD_HEADING * derivative;
  
  headingPrevError = error;
  
  return constrain(output, -0.2, 0.2);
}

// ─── Turn Rate PID ──────────────────────────────────────────────────────────

float computeTurnRatePID(float target, float current, float dt) {
  float error = target - current;
  
  error = softDeadzone(error, TURN_RATE_DEADZONE);
  
  turnRateIntegral += error * dt;
  turnRateIntegral = constrain(turnRateIntegral, -0.5, 0.5);
  
  float derivative = (error - turnRatePrevError) / dt;
  float output = KP_TURN_RATE * error + KI_TURN_RATE * turnRateIntegral + KD_TURN_RATE * derivative;
  
  turnRatePrevError = error;
  
  return constrain(output, -1.0, 1.0);
}

// ─── Motor Functions ────────────────────────────────────────────────────────

void leftISR() {
  unsigned long now = micros();
  if (now - lastLeftUS >= debounceUS) {
    lastLeftUS = now;
    leftPulses += leftFwd ? 1 : -1;
  }
}

void rightISR() {
  unsigned long now = micros();
  if (now - lastRightUS >= debounceUS) {
    lastRightUS = now;
    rightPulses += rightFwd ? 1 : -1;
  }
}

void setLeftDir(bool fwd) {
  leftFwd = fwd;
  digitalWrite(LEFT_DIR, fwd ? LEFT_FORWARD : !LEFT_FORWARD);
}

void setRightDir(bool fwd) {
  rightFwd = fwd;
  digitalWrite(RIGHT_DIR, fwd ? RIGHT_FORWARD : !RIGHT_FORWARD);
}

void setLeftPWM(int pwm) {
  leftCurrentPWM = constrain(abs(pwm), 0, MAX_PWM);
  analogWrite(LEFT_PWM, leftCurrentPWM);
}

void setRightPWM(int pwm) {
  rightCurrentPWM = constrain(abs(pwm), 0, MAX_PWM);
  analogWrite(RIGHT_PWM, rightCurrentPWM);
}

void stopMotors() {
  analogWrite(LEFT_PWM, 0);
  analogWrite(RIGHT_PWM, 0);
  leftCurrentPWM = 0;
  rightCurrentPWM = 0;
  filteredLeftSpeed = 0;
  filteredRightSpeed = 0;
  targetLinearVelocity = 0;
  targetAngularVelocity = 0;
  targetLeftSpeed = 0;
  targetRightSpeed = 0;
  
  leftIntegral = 0;
  rightIntegral = 0;
  leftPrevError = 0;
  rightPrevError = 0;
  headingIntegral = 0;
  headingPrevError = 0;
  turnRateIntegral = 0;
  turnRatePrevError = 0;
  turnRateControlActive = false;
}

// ─── Speed to PWM ──────────────────────────────────────────────────────────

float speedToPWM(float speed, bool isLeft) {
  float pwm;
  
  if (isLeft) {
    if (speed >= 0) {
      pwm = (speed - LEFT_FWD_INTERCEPT) / LEFT_FWD_SLOPE;
    } else {
      float absSpeed = -speed;
      pwm = (absSpeed - LEFT_REV_INTERCEPT) / LEFT_REV_SLOPE;
      pwm *= LEFT_REV_COMP;
      pwm = -pwm;
    }
  } else {
    if (speed >= 0) {
      pwm = (speed - RIGHT_FWD_INTERCEPT) / RIGHT_FWD_SLOPE;
      pwm *= RIGHT_FWD_COMP;
    } else {
      float absSpeed = -speed;
      pwm = (absSpeed - RIGHT_REV_INTERCEPT) / RIGHT_REV_SLOPE;
      pwm = -pwm;
    }
  }
  
  if (speed > 0.01) {
    if (isLeft) {
      pwm = max(pwm, (float)LEFT_DEADZONE);
    } else {
      pwm = max(pwm, (float)RIGHT_DEADZONE);
    }
  } else if (speed < -0.01) {
    if (isLeft) {
      pwm = min(pwm, -(float)LEFT_DEADZONE);
    } else {
      pwm = min(pwm, -(float)RIGHT_DEADZONE);
    }
  } else {
    pwm = 0;
  }
  
  return constrain(pwm, -MAX_PWM, MAX_PWM);
}

// ─── Motor Speed PID ────────────────────────────────────────────────────────

float computePID_Left(float target, float current, float dt) {
  float error = target - current;
  
  if (abs(error) < 0.5) {
    leftIntegral += error * dt;
  }
  leftIntegral = constrain(leftIntegral, -MAX_PWM * 0.2, MAX_PWM * 0.2);
  
  float derivative = (error - leftPrevError) / dt;
  float output = KP_LEFT * error + KI_LEFT * leftIntegral + KD_LEFT * derivative;
  
  leftPrevError = error;
  return output;
}

float computePID_Right(float target, float current, float dt) {
  float error = target - current;
  
  if (abs(error) < 0.5) {
    rightIntegral += error * dt;
  }
  rightIntegral = constrain(rightIntegral, -MAX_PWM * 0.2, MAX_PWM * 0.2);
  
  float derivative = (error - rightPrevError) / dt;
  float output = KP_RIGHT * error + KI_RIGHT * rightIntegral + KD_RIGHT * derivative;
  
  rightPrevError = error;
  return output;
}

// ─── Main Motor Update ──────────────────────────────────────────────────────

void updateMotorSpeeds() {
  unsigned long now = micros();
  static unsigned long lastUpdateUS = 0;
  float dt = (now - lastUpdateUS) / 1000000.0f;
  lastUpdateUS = now;
  
  if (dt > 0.1) dt = 0.1;
  if (dt < 0.001) dt = 0.001;
  
  noInterrupts();
  long lp = leftPulses;
  long rp = rightPulses;
  interrupts();
  
  static long lastLeftSnap = 0;
  static long lastRightSnap = 0;
  
  float rawLeftSpeed = (lp - lastLeftSnap) * METERS_PER_PULSE / dt;
  float rawRightSpeed = (rp - lastRightSnap) * METERS_PER_PULSE / dt;
  
  lastLeftSnap = lp;
  lastRightSnap = rp;
  
  float alpha = 0.3;
  filteredLeftSpeed = alpha * rawLeftSpeed + (1 - alpha) * filteredLeftSpeed;
  filteredRightSpeed = alpha * rawRightSpeed + (1 - alpha) * filteredRightSpeed;
  
  currentLeftSpeed = filteredLeftSpeed;
  currentRightSpeed = filteredRightSpeed;
  
  // Update odometry
  updateOdometry(dt);
  
  // Update IMU at ~100Hz
  if (millis() - lastIMUUpdate > 10) {
    updateIMU();
    lastIMUUpdate = millis();
  }
  
  // Determine what correction to apply
  float headingCorrection = 0;
  float turnRateCorrection = 0;
  
  if (headingInitialized && abs(targetLinearVelocity) > 0.02 && abs(targetAngularVelocity) < 0.01) {
    headingCorrection = computeHeadingPID(targetYaw, filteredYaw, dt);
    turnRateControlActive = false;
  }
  
  if (abs(targetAngularVelocity) > 0.01) {
    turnRateControlActive = true;
    turnRateCorrection = computeTurnRatePID(targetAngularVelocity, filteredAngularVelocity, dt);
  } else {
    turnRateControlActive = false;
  }
  
  float effectiveOmega = targetAngularVelocity + headingCorrection * 0.8 + turnRateCorrection * 0.8;
  
  float vL = targetLinearVelocity - (effectiveOmega * WHEELBASE_M) / 2.0f;
  float vR = targetLinearVelocity + (effectiveOmega * WHEELBASE_M) / 2.0f;
  
  targetLeftSpeed = vL;
  targetRightSpeed = vR;
  
  setLeftDir(vL >= 0);
  setRightDir(vR >= 0);
  
  leftFeedforward = speedToPWM(targetLeftSpeed, true);
  rightFeedforward = speedToPWM(targetRightSpeed, false);
  
  leftPidCorrection = computePID_Left(targetLeftSpeed, currentLeftSpeed, dt);
  rightPidCorrection = computePID_Right(targetRightSpeed, currentRightSpeed, dt);
  
  float leftOutput = leftFeedforward + leftPidCorrection;
  float rightOutput = rightFeedforward + rightPidCorrection;
  
  static int prevLeftPWM = 0;
  static int prevRightPWM = 0;
  
  if (targetLeftSpeed != 0 || targetRightSpeed != 0) {
    int leftPWM = constrain(abs(leftOutput), 0, MAX_PWM);
    int rightPWM = constrain(abs(rightOutput), 0, MAX_PWM);
    
    int maxChange = 5;
    leftPWM = constrain(leftPWM, prevLeftPWM - maxChange, prevLeftPWM + maxChange);
    rightPWM = constrain(rightPWM, prevRightPWM - maxChange, prevRightPWM + maxChange);
    
    prevLeftPWM = leftPWM;
    prevRightPWM = rightPWM;
    
    setLeftPWM(leftPWM);
    setRightPWM(rightPWM);
  } else {
    setLeftPWM(0);
    setRightPWM(0);
    prevLeftPWM = 0;
    prevRightPWM = 0;
  }
}

// ─── Setup ──────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);

  pinMode(LEFT_DIR,  OUTPUT);
  pinMode(LEFT_PWM,  OUTPUT);
  pinMode(LEFT_SC,  INPUT_PULLUP);
  pinMode(RIGHT_DIR,  OUTPUT);
  pinMode(RIGHT_PWM,  OUTPUT);
  pinMode(RIGHT_SC,  INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(LEFT_SC), leftISR, RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_SC), rightISR, RISING);

  initMPU6050();

  setLeftDir(true);
  setRightDir(true);
  stopMotors();

  Serial.println("READY");
  Serial.println("# Commands:");
  Serial.println("#  F<speed>    - Forward with IMU correction");
  Serial.println("#  R<speed>    - Reverse with IMU correction");
  Serial.println("#  L<omega>    - Spin left with IMU rate control");
  Serial.println("#  B<omega>    - Spin right with IMU rate control");
  Serial.println("#  V<v>,<w>    - Arc turn with IMU rate control");
  Serial.println("#  C           - Calibrate IMU");
  Serial.println("#  s           - Stop");
  Serial.println("#  z           - Zero encoders and odometry");
  Serial.println("#  PING        - Check connection");
  Serial.println("# Output: CNT,time,vL,vR,linear,omega,actualOmega,yaw,x,y,leftPWM,rightPWM");
}

// ─── Main Loop ──────────────────────────────────────────────────────────────

void loop() {
  unsigned long now = millis();
  
  if (now - lastControlTime >= CONTROL_INTERVAL) {
    lastControlTime = now;
    updateMotorSpeeds();
    
    float actualOmega = (currentRightSpeed - currentLeftSpeed) / WHEELBASE_M;
    float actualLinear = (currentLeftSpeed + currentRightSpeed) / 2.0f;
    
    Serial.print("CNT,");
    Serial.print(now);
    Serial.print(',');
    Serial.print(currentLeftSpeed, 3);
    Serial.print(',');
    Serial.print(currentRightSpeed, 3);
    Serial.print(',');
    Serial.print(actualLinear, 3);
    Serial.print(',');
    Serial.print(actualOmega, 3);
    Serial.print(',');
    Serial.print(filteredAngularVelocity, 3);
    Serial.print(',');
    Serial.print(filteredYaw, 2);
    Serial.print(',');
    Serial.print(odomX, 3);
    Serial.print(',');
    Serial.print(odomY, 3);
    Serial.print(',');
    Serial.print(leftCurrentPWM);
    Serial.print(',');
    Serial.println(rightCurrentPWM);
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) return;

    if (cmd == "PING") {
      Serial.println("READY");
      
    } else if (cmd == "s" || cmd == "S") {
      stopMotors();
      Serial.println("# Stopped");
      
    } else if (cmd == "C" || cmd == "c") {
      Serial.println("# Calibrating IMU... Keep robot still!");
      mpu6050.calcGyroOffsets(true);
      mpu6050.update();
      currentYaw = mpu6050.getAngleZ();
      filteredYaw = currentYaw;
      targetYaw = currentYaw;
      previousYaw = currentYaw;
      headingIntegral = 0;
      headingPrevError = 0;
      turnRateIntegral = 0;
      turnRatePrevError = 0;
      odomYaw = filteredYaw * (PI / 180.0);
      Serial.println("# IMU Calibrated!");
      
    } else if (cmd == "z" || cmd == "Z") {
      noInterrupts();
      leftPulses = 0;
      rightPulses = 0;
      interrupts();
      // Reset odometry
      odomX = 0.0;
      odomY = 0.0;
      totalDistance = 0.0;
      Serial.println("# Counters and odometry zeroed");
      
    } else if (cmd.startsWith("F") || cmd.startsWith("f")) {
      float speed = cmd.substring(1).toFloat();
      speed = constrain(speed, 0.0, 1.2);
      targetLinearVelocity = speed;
      targetAngularVelocity = 0;
      
      if (headingInitialized) {
        targetYaw = filteredYaw;
        headingIntegral = 0;
        headingPrevError = 0;
      }
      turnRateIntegral = 0;
      turnRatePrevError = 0;
      
      leftIntegral = 0;
      rightIntegral = 0;
      leftPrevError = 0;
      rightPrevError = 0;
      Serial.print("# Forward v=");
      Serial.println(speed, 3);
      
    } else if (cmd.startsWith("R") || cmd.startsWith("r")) {
      float speed = cmd.substring(1).toFloat();
      speed = constrain(speed, 0.0, 1.2);
      targetLinearVelocity = -speed;
      targetAngularVelocity = 0;
      
      if (headingInitialized) {
        targetYaw = filteredYaw;
        headingIntegral = 0;
        headingPrevError = 0;
      }
      turnRateIntegral = 0;
      turnRatePrevError = 0;
      
      leftIntegral = 0;
      rightIntegral = 0;
      leftPrevError = 0;
      rightPrevError = 0;
      Serial.print("# Reverse v=");
      Serial.println(speed, 3);
      
    } else if (cmd.startsWith("L") || cmd.startsWith("l")) {
      float omega = cmd.substring(1).toFloat();
      omega = constrain(omega, 0.0, 2.0);
      targetLinearVelocity = 0;
      targetAngularVelocity = omega;
      turnRateIntegral = 0;
      turnRatePrevError = 0;
      leftIntegral = 0;
      rightIntegral = 0;
      leftPrevError = 0;
      rightPrevError = 0;
      Serial.print("# Spin left omega=");
      Serial.println(omega, 3);
      
    } else if (cmd.startsWith("B") || cmd.startsWith("b")) {
      float omega = cmd.substring(1).toFloat();
      omega = constrain(omega, 0.0, 2.0);
      targetLinearVelocity = 0;
      targetAngularVelocity = -omega;
      turnRateIntegral = 0;
      turnRatePrevError = 0;
      leftIntegral = 0;
      rightIntegral = 0;
      leftPrevError = 0;
      rightPrevError = 0;
      Serial.print("# Spin right omega=");
      Serial.println(omega, 3);
      
    } else if (cmd.startsWith("V") || cmd.startsWith("v")) {
      cmd = cmd.substring(1);
      int comma = cmd.indexOf(',');
      if (comma != -1) {
        float v = cmd.substring(0, comma).toFloat();
        float w = cmd.substring(comma + 1).toFloat();
        targetLinearVelocity = constrain(v, -1.2, 1.2);
        targetAngularVelocity = constrain(w, -2.0, 2.0);
        
        turnRateIntegral = 0;
        turnRatePrevError = 0;
        if (abs(v) > 0.01 && abs(w) < 0.01) {
          targetYaw = filteredYaw;
          headingIntegral = 0;
          headingPrevError = 0;
        }
        
        leftIntegral = 0;
        rightIntegral = 0;
        leftPrevError = 0;
        rightPrevError = 0;
        Serial.print("# Set v=");
        Serial.print(v, 3);
        Serial.print(", omega=");
        Serial.println(w, 3);
      }
      
    } else {
      Serial.print("# Unknown: ");
      Serial.println(cmd);
    }
  }
}