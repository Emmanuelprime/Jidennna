/**
 * diff_drive.ino  –  Differential-drive velocity controller
 * ==========================================================
 * Receives CMD_VEL:v,w  (m/s, rad/s) over serial and drives both
 * wheels at the corresponding speeds using PID + feedforward.
 *
 * Velocity limits (right motor, no-load experiments):
 *   Single wheel max : 0.65 m/s
 *   Linear max       : 0.65 m/s
 *   Angular max      : 0.65 / (L/2)  ≈  2.49 rad/s
 *
 * If CMD_VEL would exceed either wheel's limit the command is
 * uniformly scaled down while preserving the v/w ratio.
 *
 * Kinematics  (L = WHEEL_SEPARATION_M = 0.521 m):
 *   v_left  = v - w * (L/2)
 *   v_right = v + w * (L/2)
 *
 * Motor models  (from first-order system-identification experiments):
 *   Left  : v = 0.0402 * PWM - 0.3415   τ = 0.204 s
 *   Right : v = 0.0127 * PWM + 0.0059   τ = 0.219 s
 *
 * Serial protocol (115200 baud, LF-terminated):
 *   Host → ESP32
 *     PING              → READY
 *     CMD_VEL:v,w       → (silent, low-latency)
 *     STOP              → STOPPED  + motors off
 *     STREAM_ON         → STREAM_ON
 *     STREAM_OFF        → STREAM_OFF
 *     RESET_ODOM        → ODOM_RESET
 *     STATUS            → STATUS,ctrl,stream,Lkp,Lki,Lkd,Rkp,Rki,Rkd
 *     TUNE:kp,ki,kd     → PID both …
 *     TUNEL:kp,ki,kd    → PID left …
 *     TUNER:kp,ki,kd    → PID right …
 *
 *   ESP32 → Host  (20 Hz when streaming)
 *     ODOM,t_ms,l_spd,r_spd,l_pwm,r_pwm,l_sp,r_sp,v,w,x,y,yaw
 *     WATCHDOG          (if CMD_VEL timeout exceeded)
 */

#include <Arduino.h>
#include <math.h>

// ─── Pin map ──────────────────────────────────────────────────────────────────
#define LEFT_PWM      25
#define LEFT_DIR      26
#define LEFT_SC       34   // speed-counter (encoder A)

#define RIGHT_PWM     27
#define RIGHT_DIR     33
#define RIGHT_SC      35   // speed-counter (encoder A)

#define LEFT_PWM_CH    2
#define RIGHT_PWM_CH   1
#define PWM_FREQ       1000
#define PWM_RES        8
#define MAX_PWM        60

// ─── Robot kinematics ─────────────────────────────────────────────────────────
#define WHEEL_DIAMETER_M    0.165f
#define WHEEL_SEPARATION_M  0.521f
#define PULSES_PER_REV      90
#define METERS_PER_PULSE    (PI * WHEEL_DIAMETER_M / PULSES_PER_REV)
#define HALF_TRACK          (WHEEL_SEPARATION_M * 0.5f)

// ─── Velocity limits  (right-motor limited) ───────────────────────────────────
#define V_WHEEL_MAX  0.65f              // m/s  per wheel
#define V_MAX        V_WHEEL_MAX        // m/s  linear
#define W_MAX        (V_WHEEL_MAX / HALF_TRACK)  // rad/s  ≈ 2.49

// ─── Acceleration limit ──────────────────────────────────────────────────────
// Ramp setpoints at this rate to prevent jolt on startup / step changes.
// 0.4 m/s²  → reaches 0.4 m/s in 1 s;  step per 50 ms cycle = 0.02 m/s
#define MAX_ACCEL_MS2  0.4f

// ─── Timing ───────────────────────────────────────────────────────────────────
#define SPEED_CALC_US    50000UL   // 50 ms → 20 Hz speed update
#define ODOM_STREAM_MS   50UL      // 50 ms → 20 Hz telemetry
#define CMD_TIMEOUT_MS   500UL     // watchdog: stop if silent > 500 ms

// ─── Encoder debounce ─────────────────────────────────────────────────────────
#define DEBOUNCE_US  1000UL

// ─── Direction polarity ───────────────────────────────────────────────────────
#define LEFT_FORWARD   HIGH
#define RIGHT_FORWARD  LOW

// ─── PID structure ────────────────────────────────────────────────────────────
struct PIDController {
  float Kp, Ki, Kd;
  float setpoint;
  float integral;
  float last_error;
  float output;
  float integral_limit;
  float output_limit;
  unsigned long last_update_us;
};

// ─── Globals ──────────────────────────────────────────────────────────────────
volatile long          leftPulses  = 0;
volatile long          rightPulses = 0;
volatile unsigned long lastLeftUS  = 0;
volatile unsigned long lastRightUS = 0;
volatile bool          leftFwd     = true;
volatile bool          rightFwd    = true;

float leftSpeed  = 0.0f;
float rightSpeed = 0.0f;
int   leftPWM    = 0;
int   rightPWM   = 0;

unsigned long lastSpeedCalcUS = 0;
long lastLeftPulses  = 0;
long lastRightPulses = 0;

// Odometry (dead-reckoning from encoders)
float odomX   = 0.0f;
float odomY   = 0.0f;
float odomYaw = 0.0f;
float odomV   = 0.0f;
float odomW   = 0.0f;

PIDController leftPID;
PIDController rightPID;

// Commanded targets — PID setpoints ramp toward these each speed cycle
float leftTargetSP  = 0.0f;
float rightTargetSP = 0.0f;

bool          streaming        = false;
bool          speedCtrlEnabled = false;
unsigned long lastStreamMS     = 0;
unsigned long lastCmdVelMS     = 0;

// ─── ISRs ─────────────────────────────────────────────────────────────────────
void IRAM_ATTR leftISR() {
  unsigned long now = micros();
  if (now - lastLeftUS >= DEBOUNCE_US) {
    lastLeftUS = now;
    leftPulses += leftFwd ? 1 : -1;
  }
}

void IRAM_ATTR rightISR() {
  unsigned long now = micros();
  if (now - lastRightUS >= DEBOUNCE_US) {
    lastRightUS = now;
    rightPulses += rightFwd ? 1 : -1;
  }
}

// ─── PID ──────────────────────────────────────────────────────────────────────
void pidInit(PIDController& pid, float kp, float ki, float kd,
             float ilim, float olim) {
  pid.Kp = kp; pid.Ki = ki; pid.Kd = kd;
  pid.setpoint = pid.integral = pid.last_error = pid.output = 0.0f;
  pid.integral_limit = ilim;
  pid.output_limit   = olim;
  pid.last_update_us = micros();
}

float pidUpdate(PIDController& pid, float measured) {
  unsigned long now = micros();
  float dt = (now - pid.last_update_us) * 1e-6f;
  if (dt < 0.001f) return pid.output;
  pid.last_update_us = now;

  float error = pid.setpoint - measured;

  pid.integral = constrain(pid.integral + error * dt,
                           -pid.integral_limit, pid.integral_limit);

  float deriv = (error - pid.last_error) / dt;
  pid.last_error = error;

  pid.output = constrain(pid.Kp * error + pid.Ki * pid.integral + pid.Kd * deriv,
                         -pid.output_limit, pid.output_limit);
  return pid.output;
}

// ─── Feedforward (inverted linear steady-state model) ─────────────────────────
// Clamp to 0 when |setpoint| < dead-zone so motors stop cleanly at v=0.
inline float leftFF(float v) {
  return (fabsf(v) > 0.05f) ? (v + 0.3415f) / 0.0402f : 0.0f;
}
inline float rightFF(float v) {
  return (fabsf(v) > 0.05f) ? (v - 0.0059f) / 0.0127f : 0.0f;
}

// ─── Low-level motor helpers ──────────────────────────────────────────────────
void setLeftDir(bool fwd) {
  digitalWrite(LEFT_DIR, fwd ? LEFT_FORWARD : !LEFT_FORWARD);
  leftFwd = fwd;
}
void setRightDir(bool fwd) {
  digitalWrite(RIGHT_DIR, fwd ? RIGHT_FORWARD : !RIGHT_FORWARD);
  rightFwd = fwd;
}

void applyLeftPWM(int pwm) {
  pwm = constrain(abs(pwm), 0, MAX_PWM);
  leftPWM = pwm;
  ledcWrite(LEFT_PWM_CH, pwm);
}
void applyRightPWM(int pwm) {
  pwm = constrain(abs(pwm), 0, MAX_PWM);
  rightPWM = pwm;
  ledcWrite(RIGHT_PWM_CH, pwm);
}

void stopMotors() {
  speedCtrlEnabled    = false;
  leftTargetSP        = 0.0f;
  rightTargetSP       = 0.0f;
  leftPID.setpoint    = 0.0f;
  rightPID.setpoint   = 0.0f;
  leftPID.integral    = 0.0f;
  rightPID.integral   = 0.0f;
  applyLeftPWM(0);
  applyRightPWM(0);
  setLeftDir(true);
  setRightDir(true);
}

// ─── Differential drive ───────────────────────────────────────────────────────

/**
 * Scale (v, w) uniformly so neither wheel exceeds V_WHEEL_MAX.
 * The v/w ratio is preserved (the robot still curves the same way,
 * just slower).
 */
void scaleCmdVel(float v, float w, float& vs, float& ws) {
  float vL = v - w * HALF_TRACK;
  float vR = v + w * HALF_TRACK;
  float peak = max(fabsf(vL), fabsf(vR));
  if (peak > V_WHEEL_MAX) {
    float s = V_WHEEL_MAX / peak;
    vs = v * s;
    ws = w * s;
  } else {
    vs = v;
    ws = w;
  }
}

void setCmdVel(float v, float w) {
  float vs, ws;
  scaleCmdVel(v, w, vs, ws);

  float vL = vs - ws * HALF_TRACK;
  float vR = vs + ws * HALF_TRACK;

  // Reset integral on direction reversal
  if ((vL >= 0.0f) != leftFwd)  leftPID.integral  = 0.0f;
  if ((vR >= 0.0f) != rightFwd) rightPID.integral = 0.0f;

  setLeftDir(vL  >= 0.0f);
  setRightDir(vR >= 0.0f);

  // Store targets; updateSpeeds() ramps the PID setpoints toward these
  leftTargetSP  = vL;
  rightTargetSP = vR;

  speedCtrlEnabled = true;
  lastCmdVelMS     = millis();
}

// ─── Speed + odometry update ──────────────────────────────────────────────────
void updateSpeeds() {
  unsigned long now = micros();
  if (now - lastSpeedCalcUS < SPEED_CALC_US) return;

  noInterrupts();
  long lp = leftPulses;
  long rp = rightPulses;
  interrupts();

  float dt = (now - lastSpeedCalcUS) * 1e-6f;

  leftSpeed  = (lp - lastLeftPulses)  * METERS_PER_PULSE / dt;
  rightSpeed = (rp - lastRightPulses) * METERS_PER_PULSE / dt;

  // Dead-reckoning odometry
  float v = (rightSpeed + leftSpeed) * 0.5f;
  float w = (rightSpeed - leftSpeed) / WHEEL_SEPARATION_M;
  odomV    = v;
  odomW    = w;
  odomX   += v * cosf(odomYaw) * dt;
  odomY   += v * sinf(odomYaw) * dt;
  odomYaw += w * dt;

  lastLeftPulses  = lp;
  lastRightPulses = rp;
  lastSpeedCalcUS = now;

  // ── Setpoint ramp (acceleration limit) ────────────────────────────────────
  // Move PID setpoints toward targets by at most MAX_ACCEL_MS2 * dt per cycle.
  // This prevents jolting when a new CMD_VEL arrives or on startup.
  const float max_step = MAX_ACCEL_MS2 * dt;

  float lErr = leftTargetSP  - leftPID.setpoint;
  leftPID.setpoint  += constrain(lErr, -max_step, max_step);

  float rErr = rightTargetSP - rightPID.setpoint;
  rightPID.setpoint += constrain(rErr, -max_step, max_step);
}

// ─── Speed control output ─────────────────────────────────────────────────────
void updateSpeedControl() {
  if (!speedCtrlEnabled) return;

  // Watchdog – stop if host goes silent
  if (millis() - lastCmdVelMS > CMD_TIMEOUT_MS) {
    stopMotors();
    Serial.println("WATCHDOG");
    return;
  }

  float l_pid = pidUpdate(leftPID,  leftSpeed);
  float r_pid = pidUpdate(rightPID, rightSpeed);

  int l_cmd = constrain((int)roundf(leftFF(leftPID.setpoint)   + l_pid), 0, MAX_PWM);
  int r_cmd = constrain((int)roundf(rightFF(rightPID.setpoint) + r_pid), 0, MAX_PWM);

  applyLeftPWM(l_cmd);
  applyRightPWM(r_cmd);
}

// ─── Telemetry ────────────────────────────────────────────────────────────────
void streamOdom() {
  if (!streaming) return;
  unsigned long now = millis();
  if (now - lastStreamMS < ODOM_STREAM_MS) return;
  lastStreamMS = now;

  // ODOM,t_ms,l_spd,r_spd,l_pwm,r_pwm,l_sp,r_sp,v,w,x,y,yaw
  Serial.print("ODOM,");
  Serial.print(now);                    Serial.print(',');
  Serial.print(leftSpeed,  3);          Serial.print(',');
  Serial.print(rightSpeed, 3);          Serial.print(',');
  Serial.print(leftPWM);                Serial.print(',');
  Serial.print(rightPWM);               Serial.print(',');
  Serial.print(leftPID.setpoint,  3);   Serial.print(',');
  Serial.print(rightPID.setpoint, 3);   Serial.print(',');
  Serial.print(odomV, 3);               Serial.print(',');
  Serial.print(odomW, 3);               Serial.print(',');
  Serial.print(odomX, 4);               Serial.print(',');
  Serial.print(odomY, 4);               Serial.print(',');
  Serial.println(odomYaw, 4);
}

// ─── Command parser ───────────────────────────────────────────────────────────
bool parsePID(const String& s, float& kp, float& ki, float& kd) {
  int c1 = s.indexOf(',');
  int c2 = s.indexOf(',', c1 + 1);
  if (c1 < 1 || c2 < 1) return false;
  kp = s.substring(0, c1).toFloat();
  ki = s.substring(c1 + 1, c2).toFloat();
  kd = s.substring(c2 + 1).toFloat();
  return true;
}

void handleCommand(const String& cmd) {

  if (cmd == "PING") {
    Serial.println("READY");

  } else if (cmd == "STOP") {
    stopMotors();
    streaming = false;
    Serial.println("STOPPED");

  } else if (cmd == "STREAM_ON") {
    streaming    = true;
    lastStreamMS = millis();
    Serial.println("STREAM_ON");

  } else if (cmd == "STREAM_OFF") {
    streaming = false;
    Serial.println("STREAM_OFF");

  } else if (cmd == "RESET_ODOM") {
    odomX = odomY = odomYaw = 0.0f;
    Serial.println("ODOM_RESET");

  } else if (cmd == "STATUS") {
    Serial.print("STATUS,");
    Serial.print(speedCtrlEnabled ? 1 : 0); Serial.print(',');
    Serial.print(streaming        ? 1 : 0); Serial.print(',');
    Serial.print(leftPID.Kp,  2);           Serial.print(',');
    Serial.print(leftPID.Ki,  2);           Serial.print(',');
    Serial.print(leftPID.Kd,  2);           Serial.print(',');
    Serial.print(rightPID.Kp, 2);           Serial.print(',');
    Serial.print(rightPID.Ki, 2);           Serial.print(',');
    Serial.println(rightPID.Kd, 2);

  } else if (cmd.startsWith("CMD_VEL:")) {
    int comma = cmd.indexOf(',', 8);
    if (comma > 8) {
      float v = cmd.substring(8, comma).toFloat();
      float w = cmd.substring(comma + 1).toFloat();
      setCmdVel(v, w);
      // No verbose ACK – keep command-loop latency minimal
    }

  } else if (cmd.startsWith("TUNE:")) {
    float kp, ki, kd;
    if (parsePID(cmd.substring(5), kp, ki, kd)) {
      pidInit(leftPID,  kp, ki, kd, 100.0f, 30.0f);
      pidInit(rightPID, kp, ki, kd, 100.0f, 30.0f);
      Serial.print("PID both Kp=");  Serial.print(kp, 2);
      Serial.print(" Ki=");           Serial.print(ki, 2);
      Serial.print(" Kd=");           Serial.println(kd, 2);
    }

  } else if (cmd.startsWith("TUNEL:")) {
    float kp, ki, kd;
    if (parsePID(cmd.substring(6), kp, ki, kd)) {
      pidInit(leftPID, kp, ki, kd, 100.0f, 30.0f);
      Serial.print("PID left Kp="); Serial.print(kp, 2);
      Serial.print(" Ki=");          Serial.print(ki, 2);
      Serial.print(" Kd=");          Serial.println(kd, 2);
    }

  } else if (cmd.startsWith("TUNER:")) {
    float kp, ki, kd;
    if (parsePID(cmd.substring(6), kp, ki, kd)) {
      pidInit(rightPID, kp, ki, kd, 100.0f, 30.0f);
      Serial.print("PID right Kp="); Serial.print(kp, 2);
      Serial.print(" Ki=");           Serial.print(ki, 2);
      Serial.print(" Kd=");           Serial.println(kd, 2);
    }

  } else {
    Serial.print("UNKNOWN:");
    Serial.println(cmd);
  }
}

// ─── Setup / Loop ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(LEFT_DIR,  OUTPUT);
  pinMode(RIGHT_DIR, OUTPUT);
  pinMode(LEFT_SC,   INPUT_PULLUP);
  pinMode(RIGHT_SC,  INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(LEFT_SC),  leftISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_SC), rightISR, RISING);

  ledcSetup(LEFT_PWM_CH,  PWM_FREQ, PWM_RES);
  ledcSetup(RIGHT_PWM_CH, PWM_FREQ, PWM_RES);
  ledcAttachPin(LEFT_PWM,  LEFT_PWM_CH);
  ledcAttachPin(RIGHT_PWM, RIGHT_PWM_CH);

  // Initial PID gains
  // Left  motor: Ki=2 works well (slight overshoot, corrects quickly)
  // Right motor: Ki=5 needed — right motor has lower gain (K=0.0127 vs 0.0402)
  //              and undershot by ~0.02 m/s at 0.2 m/s with Ki=2
  pidInit(leftPID,  10.0f, 2.0f, 0.1f, 100.0f, 30.0f);
  pidInit(rightPID, 10.0f, 5.0f, 0.1f, 100.0f, 30.0f);

  setLeftDir(true);
  setRightDir(true);
  applyLeftPWM(0);
  applyRightPWM(0);

  lastSpeedCalcUS = micros();
  lastCmdVelMS    = millis();

  Serial.println("READY");
  // Broadcast limits so host knows the operating envelope
  Serial.print("LIMITS,V_MAX=");  Serial.print(V_MAX,  3);
  Serial.print(",W_MAX=");         Serial.print(W_MAX,  3);
  Serial.print(",L=");             Serial.println(WHEEL_SEPARATION_M, 3);
}

void loop() {
  updateSpeeds();
  updateSpeedControl();
  streamOdom();

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() > 0) {
      handleCommand(cmd);
    }
  }
}
