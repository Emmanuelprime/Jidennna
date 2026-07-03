/**
 * diff_drive_imu.ino  –  Differential-drive controller with MPU6050 IMU
 * ======================================================================
 * Extends diff_drive.ino with a gyroscope-based yaw-rate feedback loop
 * that corrects straight-line drift and improves turn accuracy.
 *
 * ── CONTROL ARCHITECTURE (three layers) ──────────────────────────────
 *
 *  CMD_VEL:v,w  (host, 10 Hz)
 *       │
 *       ▼
 *  ┌─────────────────────────────────────────────────────────────────┐
 *  │  LAYER 1 – Yaw-rate controller  (100 Hz, MPU6050 gyro Z-axis)  │
 *  │                                                                 │
 *  │   error   = w_cmd  –  ω_gyro                                   │
 *  │   w_eff   = w_cmd  +  Kp·error  +  Ki·∫error                   │
 *  │                                                                 │
 *  │   When w_cmd=0: holds heading against drift.                    │
 *  │   When w_cmd≠0: tracks the commanded turn rate precisely.       │
 *  └────────────────────────────────┬────────────────────────────────┘
 *                                   │ w_eff  (100 Hz)
 *                                   ▼
 *  ┌─────────────────────────────────────────────────────────────────┐
 *  │  LAYER 2 – Diff-drive kinematics                                │
 *  │                                                                 │
 *  │   v_left  = v_cmd – w_eff × (L/2)                              │
 *  │   v_right = v_cmd + w_eff × (L/2)                              │
 *  └────────────────────────────────┬────────────────────────────────┘
 *                                   │ v_left, v_right  (setpoint targets)
 *                                   ▼
 *  ┌─────────────────────────────────────────────────────────────────┐
 *  │  LAYER 3 – Wheel speed PID  (5 Hz, encoder feedback)           │
 *  │                                                                 │
 *  │   Setpoint ramp  (20 Hz, 0.6 m/s²)                             │
 *  │   FF + PID  →  PWM target                                       │
 *  │   Slew rate limiter  (300 PWM/s)  →  motor                     │
 *  └─────────────────────────────────────────────────────────────────┘
 *
 * ── HARDWARE ──────────────────────────────────────────────────────
 *  MPU6050 → ESP32
 *    VCC → 3.3 V     SDA → GPIO 21
 *    GND → GND       SCL → GPIO 22
 *    AD0 → GND       (I2C address 0x68)
 *
 *  Mounting: chip flat, Z-axis pointing UP.
 *  If the robot spins the wrong way under yaw correction, set
 *  GYRO_SIGN to -1 (or send  GYRO_SIGN:-1  over serial).
 *
 * ── SERIAL PROTOCOL  (115200 baud, LF-terminated) ──────────────────
 *  Host → ESP32  (same as diff_drive.ino, plus:)
 *    YAW_TUNE:kp,ki   – set yaw-rate controller gains
 *    GYRO_SIGN:1      – set gyro sign (+1 or -1)
 *    RECALIBRATE      – re-run gyro bias calibration (keep robot still)
 *
 *  ESP32 → Host  (20 Hz when streaming)
 *    ODOM,t_ms,l_spd,r_spd,l_pwm,r_pwm,l_sp,r_sp,v,w_gyro,x,y,yaw
 *    IMU,t_ms,w_gyro,yaw_gyro       – raw gyro data at 20 Hz
 *    WATCHDOG
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

// ─── Pin map ──────────────────────────────────────────────────────────────────
#define LEFT_PWM      25
#define LEFT_DIR      26
#define LEFT_SC       34   // single-channel encoder (no internal pull-up on ESP32)

#define RIGHT_PWM     27
#define RIGHT_DIR     33
#define RIGHT_SC      35   // same limitation

#define LEFT_PWM_CH    2
#define RIGHT_PWM_CH   1
#define PWM_FREQ       1000
#define PWM_RES        8
#define MAX_PWM        60

#define I2C_SDA  21
#define I2C_SCL  22

// ─── Robot kinematics ─────────────────────────────────────────────────────────
#define WHEEL_DIAMETER_M    0.165f
#define WHEEL_SEPARATION_M  0.521f
#define PULSES_PER_REV      90
#define METERS_PER_PULSE    (PI * WHEEL_DIAMETER_M / PULSES_PER_REV)
#define HALF_TRACK          (WHEEL_SEPARATION_M * 0.5f)

// ─── Velocity limits  (right-motor limited, no-load) ─────────────────────────
#define V_WHEEL_MAX  0.65f
#define V_MAX        V_WHEEL_MAX
#define W_MAX        (V_WHEEL_MAX / HALF_TRACK)   // ≈ 2.49 rad/s

// ─── Motor dead-zone thresholds ───────────────────────────────────────────────
// Minimum PWM that produces measurable motion.  Prevents limit-cycle stall.
#define LEFT_MIN_PWM   9
#define RIGHT_MIN_PWM  2

// ─── Right motor load compensation ───────────────────────────────────────────
// The no-load model underestimates right motor PWM under the robot's weight.
// Start at 1.25 and tune until the robot drives straight at v=0.2, w=0.
// You can also adjust live via TUNER:kp,ki,kd  or re-run the load experiments.
#define RIGHT_FF_LOAD_FACTOR  1.25f

// ─── Control timing ───────────────────────────────────────────────────────────
#define SPEED_CALC_US    200000UL  // 200 ms → 5 Hz  speed measurement window
#define RAMP_UPDATE_US    50000UL  // 50 ms  → 20 Hz setpoint ramp
#define IMU_UPDATE_US     10000UL  // 10 ms  → 100 Hz yaw controller
#define ODOM_STREAM_MS      50UL  // 50 ms  → 20 Hz telemetry
#define CMD_TIMEOUT_MS     500UL  // watchdog
#define CTRL_MIN_US        1000UL  // PWM loop cap: 1 kHz

// ─── Encoder debounce ────────────────────────────────────────────────────────
// 2 ms: rejects PWM switching noise on floating GPIO 34/35.
// Safe at max speed: pulse interval = 0.0057/0.65 = 8.8 ms >> 2 ms.
#define DEBOUNCE_US  2000UL

// ─── MPU6050 (Adafruit library) ───────────────────────────────────────────────
// No raw register addresses needed — the library handles all I2C details.
// Gyro range    : ±250 deg/s  (robot max ≈ 143 deg/s)
// Filter bw     : 21 Hz       (smooths vibration noise)

// Flip to -1 if the robot over-corrects in the wrong direction.
// Can also be changed at runtime via  GYRO_SIGN:-1  over serial.
#define GYRO_SIGN_DEFAULT  1

// Number of still samples averaged to calibrate the gyro zero-rate offset.
#define GYRO_CAL_SAMPLES  500

// Software EMA on the gyro Z reading (on top of the chip's 21 Hz hardware LPF).
// α = 0.4 → τ ≈ 15 ms at 100 Hz.  Increase toward 1.0 for less filtering.
// Only needed if you observe high-frequency oscillation in yaw correction.
#define GYRO_FILTER_ALPHA  0.4f

// ─── Direction polarity ───────────────────────────────────────────────────────
#define LEFT_FORWARD   HIGH
#define RIGHT_FORWARD  LOW

// ─── PID structure ────────────────────────────────────────────────────────────
struct PIDController {
  float Kp, Ki, Kd;
  float setpoint;
  float integral;
  float last_measured;     // for derivative-on-measurement
  float filtered_deriv;
  float output;
  float integral_limit;
  float output_limit;
  unsigned long last_update_us;
};

// ─── Yaw-rate controller ──────────────────────────────────────────────────────
struct YawController {
  float Kp;            // rad/s correction per rad/s error
  float Ki;
  float integral;
  float integral_limit;
  float w_cmd;         // commanded yaw rate (rad/s, from CMD_VEL)
  float w_gyro;        // measured yaw rate (rad/s, from IMU)
  float w_effective;   // corrected yaw rate fed to kinematics
};

// ─── Globals ──────────────────────────────────────────────────────────────────

// Encoder state
volatile long          leftPulses  = 0;
volatile long          rightPulses = 0;
volatile unsigned long lastLeftUS  = 0;
volatile unsigned long lastRightUS = 0;
volatile bool          leftFwd     = true;
volatile bool          rightFwd    = true;

// Wheel speed (EMA filtered, signed m/s)
float leftSpeed  = 0.0f;
float rightSpeed = 0.0f;
int   leftPWM    = 0;
int   rightPWM   = 0;

unsigned long lastSpeedCalcUS = 0;
long lastLeftPulses  = 0;
long lastRightPulses = 0;

// Odometry (encoder dead-reckoning, corrected by gyro yaw)
float odomX      = 0.0f;
float odomY      = 0.0f;
float odomYaw    = 0.0f;   // integrated from gyro
float odomV      = 0.0f;
float odomW      = 0.0f;   // gyro yaw rate

// PID controllers
PIDController leftPID;
PIDController rightPID;

// Yaw controller
YawController yawCtrl;

// Gyro
Adafruit_MPU6050 mpu;            // Adafruit library object
float  gyroBiasZ      = 0.0f;    // rad/s bias calibrated at startup
float  gyroFiltered   = 0.0f;    // EMA-filtered gyro Z output (rad/s)
int    gyroSign       = GYRO_SIGN_DEFAULT;
float  yawGyroDeg = 0.0f;   // integrated yaw from gyro (degrees, for reference)

// CMD_VEL storage for yaw controller to recompute targets
float vCmd = 0.0f;
float wCmd = 0.0f;

// Setpoint targets
float leftTargetSP  = 0.0f;
float rightTargetSP = 0.0f;

// Slewed PWM output
float leftPWMSlewed  = 0.0f;
float rightPWMSlewed = 0.0f;

// Control flags
bool  streaming             = false;
bool  speedCtrlEnabled      = false;
bool  speedUpdatedThisCycle = false;

unsigned long lastStreamMS  = 0;
unsigned long lastCmdVelMS  = 0;
unsigned long lastCtrlUS    = 0;
unsigned long lastRampUS    = 0;
unsigned long lastImuUS     = 0;

// ─── Encoder ISRs ────────────────────────────────────────────────────────────
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
  pid.Kp = kp;  pid.Ki = ki;  pid.Kd = kd;
  pid.setpoint = pid.integral = pid.last_measured = 0.0f;
  pid.filtered_deriv = pid.output = 0.0f;
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
  float P     = pid.Kp * error;

  // Derivative on measurement (avoids kick on setpoint steps)
  float raw_deriv    = -(measured - pid.last_measured) / dt;
  pid.filtered_deriv = 0.3f * raw_deriv + 0.7f * pid.filtered_deriv;
  float D            = pid.Kd * pid.filtered_deriv;
  pid.last_measured  = measured;

  // Clamping anti-windup
  float I       = pid.Ki * pid.integral;
  float est     = P + I + D;
  bool  pos_sat = (est >=  pid.output_limit);
  bool  neg_sat = (est <= -pid.output_limit);
  bool  windup  = (pos_sat && error > 0.0f) || (neg_sat && error < 0.0f);
  if (!windup) {
    pid.integral = constrain(pid.integral + error * dt,
                             -pid.integral_limit, pid.integral_limit);
  }

  pid.output = constrain(P + pid.Ki * pid.integral + D,
                         -pid.output_limit, pid.output_limit);
  return pid.output;
}

// ─── Yaw controller ───────────────────────────────────────────────────────────

void yawCtrlInit(YawController& y, float kp, float ki, float ilim) {
  y.Kp = kp;  y.Ki = ki;
  y.integral       = 0.0f;
  y.integral_limit = ilim;
  y.w_cmd = y.w_gyro = y.w_effective = 0.0f;
}

void yawCtrlUpdate(YawController& y, float w_gyro_measured, float dt) {
  y.w_gyro = w_gyro_measured;

  float error  = y.w_cmd - w_gyro_measured;

  // Anti-windup: clamp integral
  y.integral = constrain(y.integral + error * dt,
                         -y.integral_limit, y.integral_limit);

  float correction = y.Kp * error + y.Ki * y.integral;

  // Limit how much the yaw controller can change the effective w
  // (cannot correct more than ±50 % of W_MAX)
  correction = constrain(correction, -W_MAX * 0.5f, W_MAX * 0.5f);

  // Effective w used by kinematics
  y.w_effective = constrain(y.w_cmd + correction, -W_MAX, W_MAX);
}

void yawCtrlReset(YawController& y) {
  y.integral    = 0.0f;
  y.w_effective = y.w_cmd;
}

// ─── MPU6050  (via Adafruit library) ────────────────────────────────────────

/** Initialise MPU6050: configure gyro range and low-pass filter. */
bool mpuInit() {
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!mpu.begin()) return false;

  mpu.setGyroRange(MPU6050_RANGE_250_DEG);     // ±250 deg/s
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);  // 21 Hz LPF on sensor output
  return true;
}

/**
 * Calibrate gyro Z-axis bias.
 * Keep the robot COMPLETELY STILL during calibration.
 * Averages GYRO_CAL_SAMPLES readings to find the zero-rate offset.
 */
void mpuCalibrate() {
  Serial.println("CALIBRATING – keep robot still …");
  double sum = 0.0;
  sensors_event_t a, g, temp;
  for (int i = 0; i < GYRO_CAL_SAMPLES; i++) {
    mpu.getEvent(&a, &g, &temp);
    sum += g.gyro.z;   // already in rad/s
    delay(2);
  }
  gyroBiasZ = (float)(sum / GYRO_CAL_SAMPLES);
  Serial.print("CAL_DONE,bias_rad_s=");
  Serial.println(gyroBiasZ, 6);
}

/**
 * Read calibrated, filtered Z-axis yaw rate in rad/s.
 * Positive = counter-clockwise (left turn) when GYRO_SIGN = +1.
 * Applies bias subtraction then a software EMA on top of the chip's
 * built-in 21 Hz hardware low-pass filter.
 */
float mpuReadYawRate() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  float raw = (float)gyroSign * (g.gyro.z - gyroBiasZ);
  gyroFiltered = GYRO_FILTER_ALPHA * raw
               + (1.0f - GYRO_FILTER_ALPHA) * gyroFiltered;
  return gyroFiltered;
}

// ─── Feedforward ──────────────────────────────────────────────────────────────
// Models identified under no-load.  rightFF is scaled by RIGHT_FF_LOAD_FACTOR
// to compensate for the right motor running slower under the robot's weight.

inline float leftFF(float v) {
  float s = fabsf(v);
  return (s > 0.05f) ? (s + 0.3415f) / 0.0402f : 0.0f;
}

inline float rightFF(float v) {
  float s = fabsf(v);
  return (s > 0.05f) ? (s - 0.0059f) / 0.0127f * RIGHT_FF_LOAD_FACTOR : 0.0f;
}

// ─── Motor helpers ────────────────────────────────────────────────────────────

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

// ─── Stop ─────────────────────────────────────────────────────────────────────

void stopMotors() {
  speedCtrlEnabled   = false;
  vCmd = wCmd        = 0.0f;
  leftTargetSP       = 0.0f;
  rightTargetSP      = 0.0f;
  leftPID.setpoint   = 0.0f;
  rightPID.setpoint  = 0.0f;
  leftPID.integral   = 0.0f;
  rightPID.integral  = 0.0f;
  leftPWMSlewed      = 0.0f;
  rightPWMSlewed     = 0.0f;
  yawCtrlReset(yawCtrl);
  applyLeftPWM(0);
  applyRightPWM(0);
  setLeftDir(true);
  setRightDir(true);
}

// ─── Differential drive kinematics ───────────────────────────────────────────

/** Scale (v,w) so neither wheel exceeds V_WHEEL_MAX, preserving v/w ratio. */
void scaleCmdVel(float v, float w, float& vs, float& ws) {
  float vL   = v - w * HALF_TRACK;
  float vR   = v + w * HALF_TRACK;
  float peak = max(fabsf(vL), fabsf(vR));
  if (peak > V_WHEEL_MAX) { float s = V_WHEEL_MAX / peak; vs = v*s; ws = w*s; }
  else                    { vs = v; ws = w; }
}

/** Compute wheel speed targets from (v, w_effective) and apply direction resets. */
void computeWheelTargets(float v, float w) {
  float vs, ws;
  scaleCmdVel(v, w, vs, ws);
  float vL = vs - ws * HALF_TRACK;
  float vR = vs + ws * HALF_TRACK;

  // On direction reversal: full PID state reset (see diff_drive.ino comments)
  if ((vL >= 0.0f) != leftFwd) {
    leftPID.integral = leftPID.setpoint = leftPID.last_measured = 0.0f;
    leftPID.filtered_deriv = 0.0f;
    leftSpeed = 0.0f;
    leftPWMSlewed = 0.0f;
  }
  if ((vR >= 0.0f) != rightFwd) {
    rightPID.integral = rightPID.setpoint = rightPID.last_measured = 0.0f;
    rightPID.filtered_deriv = 0.0f;
    rightSpeed = 0.0f;
    rightPWMSlewed = 0.0f;
  }
  setLeftDir(vL  >= 0.0f);
  setRightDir(vR >= 0.0f);

  leftTargetSP  = vL;
  rightTargetSP = vR;
}

/**
 * Store a new CMD_VEL command.
 * The yaw controller uses the stored w_cmd; targets are recomputed by
 * updateIMU() at 100 Hz using the corrected w_effective.
 */
void setCmdVel(float v, float w) {
  float vs, ws;
  scaleCmdVel(v, w, vs, ws);

  // Reset yaw integral if angular command changes significantly
  if (fabsf(ws - wCmd) > 0.3f) yawCtrlReset(yawCtrl);

  vCmd = vs;
  wCmd = ws;
  yawCtrl.w_cmd = ws;

  computeWheelTargets(vCmd, yawCtrl.w_effective);

  speedCtrlEnabled = true;
  lastCmdVelMS     = millis();
}

// ─── IMU update  (100 Hz) ────────────────────────────────────────────────────

void updateIMU() {
  unsigned long now = micros();
  if (now - lastImuUS < IMU_UPDATE_US) return;
  float dt    = (now - lastImuUS) * 1e-6f;
  lastImuUS   = now;

  // Read gyro and update yaw controller
  float w_gyro = mpuReadYawRate();
  yawCtrlUpdate(yawCtrl, w_gyro, dt);

  // Integrate gyro yaw for odometry (more accurate than encoder-estimated yaw)
  odomYaw  += w_gyro * dt;
  odomW     = w_gyro;
  yawGyroDeg += w_gyro * (180.0f / PI) * dt;

  // Recompute wheel speed targets using corrected w_effective
  if (speedCtrlEnabled) {
    computeWheelTargets(vCmd, yawCtrl.w_effective);
  }
}

// ─── Speed + odometry update  (5 Hz) ─────────────────────────────────────────

void updateSpeeds() {
  unsigned long now = micros();
  if (now - lastSpeedCalcUS < SPEED_CALC_US) return;

  noInterrupts();
  long lp = leftPulses;
  long rp = rightPulses;
  interrupts();

  float dt  = (now - lastSpeedCalcUS) * 1e-6f;
  float rawL = (lp - lastLeftPulses)  * METERS_PER_PULSE / dt;
  float rawR = (rp - lastRightPulses) * METERS_PER_PULSE / dt;

  // Outlier rejection: discard if speed jumps > 0.35 m/s (likely noise spike)
  if (fabsf(rawL - leftSpeed)  > 0.35f) rawL = leftSpeed;
  if (fabsf(rawR - rightSpeed) > 0.35f) rawR = rightSpeed;

  // Exponential moving average (α = 0.5)
  leftSpeed  = 0.5f * rawL + 0.5f * leftSpeed;
  rightSpeed = 0.5f * rawR + 0.5f * rightSpeed;

  // Encoder-based v estimate (used with gyro yaw for odometry position)
  odomV = (rightSpeed + leftSpeed) * 0.5f;
  // Note: odomYaw and odomW are maintained by updateIMU() from gyro.
  float w_enc = (rightSpeed - leftSpeed) / WHEEL_SEPARATION_M;
  odomX += odomV * cosf(odomYaw) * dt;
  odomY += odomV * sinf(odomYaw) * dt;

  lastLeftPulses  = lp;
  lastRightPulses = rp;
  lastSpeedCalcUS = now;
  speedUpdatedThisCycle = true;
}

// ─── Setpoint ramp  (20 Hz) ───────────────────────────────────────────────────

void updateRamp() {
  unsigned long now = micros();
  if (now - lastRampUS < RAMP_UPDATE_US) return;
  float dt = (now - lastRampUS) * 1e-6f;
  lastRampUS = now;

  const float max_step = 0.6f * dt;  // MAX_ACCEL_MS2 = 0.6 m/s²
  leftPID.setpoint  += constrain(leftTargetSP  - leftPID.setpoint,  -max_step, max_step);
  rightPID.setpoint += constrain(rightTargetSP - rightPID.setpoint, -max_step, max_step);
}

// ─── Wheel speed control  (~1 kHz) ───────────────────────────────────────────

void updateSpeedControl() {
  if (!speedCtrlEnabled) return;

  if (millis() - lastCmdVelMS > CMD_TIMEOUT_MS) {
    stopMotors();
    Serial.println("WATCHDOG");
    return;
  }

  unsigned long now = micros();
  float dt = (now - lastCtrlUS) * 1e-6f;
  if (dt < 0.001f) return;
  lastCtrlUS = now;

  // PID: only on fresh measurement (5 Hz)
  if (speedUpdatedThisCycle) {
    speedUpdatedThisCycle = false;

    float l_pid = pidUpdate(leftPID,  leftSpeed);
    float r_pid = pidUpdate(rightPID, rightSpeed);

    // Sign correction for backward motion (see diff_drive.ino)
    if (leftPID.setpoint  < 0.0f) l_pid = -l_pid;
    if (rightPID.setpoint < 0.0f) r_pid = -r_pid;

    leftPWMSlewed  = leftFF(leftPID.setpoint)   + l_pid;
    rightPWMSlewed = rightFF(rightPID.setpoint) + r_pid;
  }

  // PWM slew-rate limiter (300 PWM/s, runs at 1 kHz)
  float max_slew = 300.0f * dt;
  float l_out = constrain((float)leftPWM  + constrain(leftPWMSlewed  - (float)leftPWM,  -max_slew, max_slew), 0.0f, (float)MAX_PWM);
  float r_out = constrain((float)rightPWM + constrain(rightPWMSlewed - (float)rightPWM, -max_slew, max_slew), 0.0f, (float)MAX_PWM);

  // Synchronized startup: advance both motors at same fractional progress
  if (leftPWMSlewed > 1.0f && rightPWMSlewed > 1.0f) {
    float l_frac   = l_out / leftPWMSlewed;
    float r_frac   = r_out / rightPWMSlewed;
    float min_frac = min(l_frac, r_frac);
    if (min_frac < 0.95f) {
      l_out = min(l_out, leftPWMSlewed  * min_frac + max_slew);
      r_out = min(r_out, rightPWMSlewed * min_frac + max_slew);
    }
  }

  // Dead zone clamp: prevent stall oscillation
  if (fabsf(leftPID.setpoint)  > 0.05f) l_out = max(l_out, (float)LEFT_MIN_PWM);
  if (fabsf(rightPID.setpoint) > 0.05f) r_out = max(r_out, (float)RIGHT_MIN_PWM);

  applyLeftPWM( (int)roundf(l_out));
  applyRightPWM((int)roundf(r_out));
}

// ─── Telemetry ────────────────────────────────────────────────────────────────

void streamTelemetry() {
  if (!streaming) return;
  unsigned long now = millis();
  if (now - lastStreamMS < ODOM_STREAM_MS) return;
  lastStreamMS = now;

  // Main ODOM packet (same format as diff_drive.ino for host compatibility)
  // w field now reports gyro yaw rate (more accurate than encoder estimate)
  Serial.print("ODOM,");
  Serial.print(now);                    Serial.print(',');
  Serial.print(leftSpeed,  3);          Serial.print(',');
  Serial.print(rightSpeed, 3);          Serial.print(',');
  Serial.print(leftPWM);                Serial.print(',');
  Serial.print(rightPWM);               Serial.print(',');
  Serial.print(leftPID.setpoint,  3);   Serial.print(',');
  Serial.print(rightPID.setpoint, 3);   Serial.print(',');
  Serial.print(odomV, 3);               Serial.print(',');
  Serial.print(odomW, 3);               Serial.print(',');  // gyro yaw rate
  Serial.print(odomX, 4);               Serial.print(',');
  Serial.print(odomY, 4);               Serial.print(',');
  Serial.println(odomYaw, 4);           // gyro-integrated yaw

  // Extra IMU packet for debugging
  Serial.print("IMU,");
  Serial.print(now);                    Serial.print(',');
  Serial.print(yawCtrl.w_gyro,    4);   Serial.print(',');
  Serial.print(yawCtrl.w_effective,4);  Serial.print(',');
  Serial.print(yawCtrl.w_cmd,     3);   Serial.print(',');
  Serial.println(yawGyroDeg,      2);
}

// ─── Command parser helpers ───────────────────────────────────────────────────

bool parsePID(const String& s, float& kp, float& ki, float& kd) {
  int c1 = s.indexOf(',');
  int c2 = s.indexOf(',', c1 + 1);
  if (c1 < 1 || c2 < 1) return false;
  kp = s.substring(0, c1).toFloat();
  ki = s.substring(c1 + 1, c2).toFloat();
  kd = s.substring(c2 + 1).toFloat();
  return true;
}

static inline float integralLimit(float ki, float olim) {
  return (ki > 0.01f) ? (olim / ki) : 10.0f;
}

// ─── Command handler ──────────────────────────────────────────────────────────

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
    yawGyroDeg = 0.0f;
    Serial.println("ODOM_RESET");

  } else if (cmd == "RECALIBRATE") {
    stopMotors();
    mpuCalibrate();

  } else if (cmd == "STATUS") {
    Serial.print("STATUS,");
    Serial.print(speedCtrlEnabled ? 1 : 0); Serial.print(',');
    Serial.print(streaming        ? 1 : 0); Serial.print(',');
    Serial.print(leftPID.Kp,  2);           Serial.print(',');
    Serial.print(leftPID.Ki,  2);           Serial.print(',');
    Serial.print(leftPID.Kd,  2);           Serial.print(',');
    Serial.print(rightPID.Kp, 2);           Serial.print(',');
    Serial.print(rightPID.Ki, 2);           Serial.print(',');
    Serial.print(rightPID.Kd, 2);           Serial.print(',');
    Serial.print(yawCtrl.Kp,  2);           Serial.print(',');
    Serial.println(yawCtrl.Ki, 2);

  } else if (cmd.startsWith("CMD_VEL:")) {
    int comma = cmd.indexOf(',', 8);
    if (comma > 8) {
      float v = cmd.substring(8, comma).toFloat();
      float w = cmd.substring(comma + 1).toFloat();
      setCmdVel(v, w);
    }

  } else if (cmd.startsWith("TUNE:")) {
    float kp, ki, kd;
    if (parsePID(cmd.substring(5), kp, ki, kd)) {
      const float olim = 15.0f;
      pidInit(leftPID,  kp, ki, kd, integralLimit(ki, olim), olim);
      pidInit(rightPID, kp, ki, kd, integralLimit(ki, olim), olim);
      Serial.print("PID both Kp="); Serial.print(kp,2);
      Serial.print(" Ki="); Serial.print(ki,2);
      Serial.print(" Kd="); Serial.println(kd,2);
    }

  } else if (cmd.startsWith("TUNEL:")) {
    float kp, ki, kd;
    if (parsePID(cmd.substring(6), kp, ki, kd)) {
      const float olim = 15.0f;
      pidInit(leftPID, kp, ki, kd, integralLimit(ki, olim), olim);
      Serial.print("PID left Kp="); Serial.print(kp,2);
      Serial.print(" Ki="); Serial.print(ki,2);
      Serial.print(" Kd="); Serial.println(kd,2);
    }

  } else if (cmd.startsWith("TUNER:")) {
    float kp, ki, kd;
    if (parsePID(cmd.substring(6), kp, ki, kd)) {
      const float olim = 15.0f;
      pidInit(rightPID, kp, ki, kd, integralLimit(ki, olim), olim);
      Serial.print("PID right Kp="); Serial.print(kp,2);
      Serial.print(" Ki="); Serial.print(ki,2);
      Serial.print(" Kd="); Serial.println(kd,2);
    }

  } else if (cmd.startsWith("YAW_TUNE:")) {
    // YAW_TUNE:kp,ki
    String params = cmd.substring(9);
    int comma     = params.indexOf(',');
    if (comma > 0) {
      float kp = params.substring(0, comma).toFloat();
      float ki = params.substring(comma + 1).toFloat();
      yawCtrl.Kp = kp;
      yawCtrl.Ki = ki;
      yawCtrlReset(yawCtrl);
      Serial.print("YAW Kp="); Serial.print(kp,3);
      Serial.print(" Ki=");    Serial.println(ki,3);
    }

  } else if (cmd.startsWith("GYRO_SIGN:")) {
    int sign = cmd.substring(10).toInt();
    if (sign == 1 || sign == -1) {
      gyroSign = sign;
      Serial.print("GYRO_SIGN="); Serial.println(gyroSign);
    }

  } else {
    Serial.print("UNKNOWN:"); Serial.println(cmd);
  }
}

// ─── Setup ────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);

  // Motor pins
  pinMode(LEFT_DIR,  OUTPUT);
  pinMode(RIGHT_DIR, OUTPUT);
  pinMode(LEFT_SC,   INPUT);   // INPUT_PULLUP silently fails on GPIO 34/35
  pinMode(RIGHT_SC,  INPUT);

  attachInterrupt(digitalPinToInterrupt(LEFT_SC),  leftISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_SC), rightISR, RISING);

  ledcSetup(LEFT_PWM_CH,  PWM_FREQ, PWM_RES);
  ledcSetup(RIGHT_PWM_CH, PWM_FREQ, PWM_RES);
  ledcAttachPin(LEFT_PWM,  LEFT_PWM_CH);
  ledcAttachPin(RIGHT_PWM, RIGHT_PWM_CH);

  setLeftDir(true);
  setRightDir(true);
  applyLeftPWM(0);
  applyRightPWM(0);

  // Wheel speed PID
  // Left  Ki=2 : integral_limit = 15/2 = 7.5
  // Right Ki=5 : integral_limit = 15/5 = 3.0
  pidInit(leftPID,  10.0f, 2.0f, 0.1f, 7.5f, 15.0f);
  pidInit(rightPID, 10.0f, 5.0f, 0.1f, 3.0f, 15.0f);

  // Yaw controller
  // Kp=2.0: aggressive enough to reject drift, small enough to avoid oscillation
  // Ki=0.5: slowly removes steady-state heading bias
  // integral_limit=1.0: limits integral contribution to ±0.5 rad/s correction
  yawCtrlInit(yawCtrl, 2.0f, 0.5f, 1.0f);

  // MPU6050
  if (!mpuInit()) {
    Serial.println("MPU6050 NOT FOUND – check wiring");
  } else {
    Serial.println("MPU6050 OK");
    mpuCalibrate();
  }

  unsigned long now = micros();
  lastSpeedCalcUS = now;
  lastCtrlUS      = now;
  lastRampUS      = now;
  lastImuUS       = now;
  lastCmdVelMS    = millis();

  Serial.println("READY");
  Serial.print("LIMITS,V_MAX="); Serial.print(V_MAX,  3);
  Serial.print(",W_MAX=");        Serial.print(W_MAX,  3);
  Serial.print(",L=");            Serial.println(WHEEL_SEPARATION_M, 3);
}

// ─── Loop ─────────────────────────────────────────────────────────────────────

void loop() {
  updateIMU();           // 100 Hz – gyro read, yaw control, target update
  updateSpeeds();        //   5 Hz – encoder speed estimation
  updateRamp();          //  20 Hz – setpoint advance
  updateSpeedControl();  //   1 kHz – PID + slew + PWM
  streamTelemetry();     //  20 Hz – serial output

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() > 0) handleCommand(cmd);
  }
}
