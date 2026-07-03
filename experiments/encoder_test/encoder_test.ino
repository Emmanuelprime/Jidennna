/**
 * encoder_test.ino
 * ================
 * Diagnostic sketch: run motors at a fixed PWM and stream raw encoder
 * counts so you can verify the encoder wiring, debounce, and pulse rates
 * before relying on the speed controller.
 *
 * Serial commands  (115200 baud, send with newline):
 *   f<pwm>    – both motors forward,  e.g. "f20"
 *   r<pwm>    – both motors reverse,  e.g. "r20"
 *   l<pwm>    – left  motor only forward
 *   ri<pwm>   – right motor only forward   (note: "ri" not "r")
 *   s         – stop all motors
 *   z         – zero (reset) pulse counters
 *   d<us>     – set debounce in microseconds, e.g. "d500"
 *   PING      – prints READY
 *
 * Output line (every 100 ms):
 *   CNT,<ms>,<leftPulses>,<rightPulses>,<leftRaw_ms>,<rightRaw_ms>
 *
 * Where leftRaw_ms / rightRaw_ms are the speeds computed over the last
 * 100 ms window (same method as the main controller but faster refresh
 * so you can watch individual pulse arrivals).
 */

#include <Arduino.h>

// ─── Pins (must match diff_drive.ino) ────────────────────────────────────────
#define LEFT_PWM      25
#define LEFT_DIR      26
#define LEFT_SC       34

#define RIGHT_PWM     27
#define RIGHT_DIR     33
#define RIGHT_SC      35

#define LEFT_PWM_CH    2
#define RIGHT_PWM_CH   1
#define PWM_FREQ       1000
#define PWM_RES        8
#define MAX_PWM        60

#define LEFT_FORWARD   HIGH
#define RIGHT_FORWARD  LOW

// ─── Kinematics ───────────────────────────────────────────────────────────────
#define WHEEL_DIAMETER_M  0.165f
#define PULSES_PER_REV    90
#define METERS_PER_PULSE  (PI * WHEEL_DIAMETER_M / PULSES_PER_REV)

// ─── Debounce (adjustable via 'd' command) ────────────────────────────────────
volatile unsigned long debounceUS = 1000UL;

// ─── Encoder state ────────────────────────────────────────────────────────────
volatile long          leftPulses  = 0;
volatile long          rightPulses = 0;
volatile unsigned long lastLeftUS  = 0;
volatile unsigned long lastRightUS = 0;
volatile bool          leftFwd     = true;
volatile bool          rightFwd    = true;

void IRAM_ATTR leftISR() {
  unsigned long now = micros();
  if (now - lastLeftUS >= debounceUS) {
    lastLeftUS = now;
    leftPulses += leftFwd ? 1 : -1;
  }
}

void IRAM_ATTR rightISR() {
  unsigned long now = micros();
  if (now - lastRightUS >= debounceUS) {
    lastRightUS = now;
    rightPulses += rightFwd ? 1 : -1;
  }
}

// ─── Motor helpers ────────────────────────────────────────────────────────────
void setLeftDir(bool fwd) {
  leftFwd = fwd;
  digitalWrite(LEFT_DIR, fwd ? LEFT_FORWARD : !LEFT_FORWARD);
}
void setRightDir(bool fwd) {
  rightFwd = fwd;
  digitalWrite(RIGHT_DIR, fwd ? RIGHT_FORWARD : !RIGHT_FORWARD);
}
void setLeftPWM(int pwm) {
  ledcWrite(LEFT_PWM_CH,  constrain(abs(pwm), 0, MAX_PWM));
}
void setRightPWM(int pwm) {
  ledcWrite(RIGHT_PWM_CH, constrain(abs(pwm), 0, MAX_PWM));
}
void stopAll() {
  setLeftPWM(0);
  setRightPWM(0);
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(LEFT_DIR,  OUTPUT);
  pinMode(RIGHT_DIR, OUTPUT);
  // Note: pins 34 & 35 are INPUT-ONLY on ESP32 – INPUT_PULLUP is ignored.
  // Add external 10 kΩ pull-ups to 3.3 V if you see noise.
  pinMode(LEFT_SC,  INPUT);
  pinMode(RIGHT_SC, INPUT);

  attachInterrupt(digitalPinToInterrupt(LEFT_SC),  leftISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_SC), rightISR, RISING);

  ledcSetup(LEFT_PWM_CH,  PWM_FREQ, PWM_RES);
  ledcSetup(RIGHT_PWM_CH, PWM_FREQ, PWM_RES);
  ledcAttachPin(LEFT_PWM,  LEFT_PWM_CH);
  ledcAttachPin(RIGHT_PWM, RIGHT_PWM_CH);

  setLeftDir(true);
  setRightDir(true);
  stopAll();

  Serial.println("READY");
  Serial.println("# Commands: f<pwm>  r<pwm>  l<pwm>  ri<pwm>  s  z  d<us>  PING");
  Serial.println("# Output:   CNT,ms,leftPulses,rightPulses,leftSpeed_ms,rightSpeed_ms");
}

// ─── Main loop ────────────────────────────────────────────────────────────────
long lastLeftSnap  = 0;
long lastRightSnap = 0;
unsigned long lastPrintMS = 0;

void loop() {
  // ── Stream encoder data every 100 ms ────────────────────────────────────
  unsigned long now = millis();
  if (now - lastPrintMS >= 100) {
    float dt = (now - lastPrintMS) / 1000.0f;
    lastPrintMS = now;

    noInterrupts();
    long lp = leftPulses;
    long rp = rightPulses;
    interrupts();

    float leftSpeed  = (lp - lastLeftSnap)  * METERS_PER_PULSE / dt;
    float rightSpeed = (rp - lastRightSnap) * METERS_PER_PULSE / dt;
    lastLeftSnap  = lp;
    lastRightSnap = rp;

    Serial.print("CNT,");
    Serial.print(now);        Serial.print(',');
    Serial.print(lp);         Serial.print(',');
    Serial.print(rp);         Serial.print(',');
    Serial.print(leftSpeed,  3); Serial.print(',');
    Serial.println(rightSpeed, 3);
  }

  // ── Serial command handler ────────────────────────────────────────────────
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) return;

    if (cmd == "PING") {
      Serial.println("READY");

    } else if (cmd == "s" || cmd == "S") {
      stopAll();
      Serial.println("# Stopped");

    } else if (cmd == "z" || cmd == "Z") {
      noInterrupts();
      leftPulses = rightPulses = 0;
      interrupts();
      lastLeftSnap = lastRightSnap = 0;
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
      setRightDir(true);
      setLeftPWM(pwm);
      setRightPWM(pwm);
      Serial.print("# Both forward PWM=");
      Serial.println(pwm);

    } else if (cmd.startsWith("ri") || cmd.startsWith("RI")) {
      // "ri" = right motor forward only
      int pwm = constrain(cmd.substring(2).toInt(), 0, MAX_PWM);
      setLeftPWM(0);
      setRightDir(true);
      setRightPWM(pwm);
      Serial.print("# Right forward PWM=");
      Serial.println(pwm);

    } else if (cmd.startsWith("r") || cmd.startsWith("R")) {
      // "r" alone = both reverse
      int pwm = constrain(cmd.substring(1).toInt(), 0, MAX_PWM);
      setLeftDir(false);
      setRightDir(false);
      setLeftPWM(pwm);
      setRightPWM(pwm);
      Serial.print("# Both reverse PWM=");
      Serial.println(pwm);

    } else if (cmd.startsWith("l") || cmd.startsWith("L")) {
      int pwm = constrain(cmd.substring(1).toInt(), 0, MAX_PWM);
      setRightPWM(0);
      setLeftDir(true);
      setLeftPWM(pwm);
      Serial.print("# Left forward PWM=");
      Serial.println(pwm);

    } else {
      Serial.print("# Unknown: ");
      Serial.println(cmd);
    }
  }
}
