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
#define SPEED_SMOOTHING  0.3f

volatile unsigned long debounceUS = 1000UL;

volatile long          leftPulses  = 0;
volatile unsigned long lastLeftUS  = 0;
volatile bool          leftFwd     = true;

volatile long          rightPulses  = 0;
volatile unsigned long lastRightUS  = 0;
volatile bool          rightFwd     = true;

int leftCurrentPWM = 0;
int rightCurrentPWM = 0;
float filteredLeftSpeed = 0;
float filteredRightSpeed = 0;

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
}

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

  setLeftDir(true);
  setRightDir(true);
  stopMotors();

  Serial.println("READY");
  Serial.println("# Commands: f<pwm>  r<pwm>  l<pwm>  b<pwm>  s  z  d<us>  PING");
  Serial.println("# Output:   CNT,ms,leftPulses,leftSpeed_mps,rightPulses,rightSpeed_mps");
}

long lastLeftSnap  = 0;
long lastRightSnap = 0;
unsigned long lastPrintMS = 0;

void loop() {
  unsigned long now = millis();
  if (now - lastPrintMS >= 100) {
    float dt = (now - lastPrintMS) / 1000.0f;
    lastPrintMS = now;

    noInterrupts();
    long lp = leftPulses;
    long rp = rightPulses;
    interrupts();

    float rawLeftSpeed = (lp - lastLeftSnap) * METERS_PER_PULSE / dt;
    float rawRightSpeed = (rp - lastRightSnap) * METERS_PER_PULSE / dt;
    
    filteredLeftSpeed = SPEED_SMOOTHING * rawLeftSpeed + (1 - SPEED_SMOOTHING) * filteredLeftSpeed;
    filteredRightSpeed = SPEED_SMOOTHING * rawRightSpeed + (1 - SPEED_SMOOTHING) * filteredRightSpeed;
    
    lastLeftSnap  = lp;
    lastRightSnap = rp;

    Serial.print("CNT,");
    Serial.print(now);
    Serial.print(',');
    Serial.print(lp);
    Serial.print(',');
    Serial.print(filteredLeftSpeed, 3);
    Serial.print(',');
    Serial.print(rp);
    Serial.print(',');
    Serial.println(filteredRightSpeed, 3);
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
    } else if (cmd == "z" || cmd == "Z") {
      noInterrupts();
      leftPulses = 0;
      rightPulses = 0;
      interrupts();
      lastLeftSnap = 0;
      lastRightSnap = 0;
      filteredLeftSpeed = 0;
      filteredRightSpeed = 0;
      Serial.println("# Counters zeroed");
    } else if (cmd.startsWith("d") || cmd.startsWith("D")) {
      unsigned long us = cmd.substring(1).toInt();
      if (us > 0) {
        debounceUS = us;
        Serial.print("# Debounce set to ");
        Serial.print(debounceUS);
        Serial.println(" us");
      }
    } else if (cmd.startsWith("f") || cmd.startsWith("F")) {
      int pwm = constrain(cmd.substring(1).toInt(), 0, MAX_PWM);
      setLeftDir(true);
      setLeftPWM(pwm);
      setRightDir(true);
      setRightPWM(pwm);
      Serial.print("# Both motors forward PWM=");
      Serial.println(pwm);
    } else if (cmd.startsWith("r") || cmd.startsWith("R")) {
      int pwm = constrain(cmd.substring(1).toInt(), 0, MAX_PWM);
      setLeftDir(false);
      setLeftPWM(pwm);
      setRightDir(false);
      setRightPWM(pwm);
      Serial.print("# Both motors reverse PWM=");
      Serial.println(pwm);
    } else if (cmd.startsWith("l") || cmd.startsWith("L")) {
      int pwm = constrain(cmd.substring(1).toInt(), 0, MAX_PWM);
      setLeftDir(false);
      setLeftPWM(pwm);
      setRightDir(true);
      setRightPWM(pwm);
      Serial.print("# Left turn PWM=");
      Serial.println(pwm);
    } else if (cmd.startsWith("b") || cmd.startsWith("B")) {
      int pwm = constrain(cmd.substring(1).toInt(), 0, MAX_PWM);
      setLeftDir(true);
      setLeftPWM(pwm);
      setRightDir(false);
      setRightPWM(pwm);
      Serial.print("# Right turn PWM=");
      Serial.println(pwm);
    } else {
      Serial.print("# Unknown: ");
      Serial.println(cmd);
    }
  }
}