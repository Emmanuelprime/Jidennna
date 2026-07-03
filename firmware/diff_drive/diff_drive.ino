/**
 * diff_drive.ino  –  Differential-drive velocity controller  (v2)
 * ================================================================
 * Receives CMD_VEL:v,w  (m/s, rad/s) and drives both wheels at the
 * corresponding speeds using feedforward + PID velocity control.
 *
 * Kinematics  (L = WHEEL_SEPARATION_M = 0.521 m):
 *   v_left  = v - w * (L/2)
 *   v_right = v + w * (L/2)
 *
 * Motor models  (first-order system-ID, no-load):
 *   Left  : v = 0.0402 * PWM - 0.3415   τ = 0.204 s
 *   Right : v = 0.0127 * PWM + 0.0059   τ = 0.219 s
 *
 * ── IMPROVEMENTS OVER v1 ──────────────────────────────────────────
 *
 * 1. Speed low-pass filter  (EMA, α = 0.4)
 *    Raw encoder counts over a 50 ms window are noisy at low speeds
 *    (1 pulse = 0.0058 m → 0.115 m/s at 20 Hz).  An exponential
 *    moving average with α = 0.4 (τ ≈ 75 ms) smooths this noise
 *    without adding excessive lag.
 *
 * 2. Derivative-on-measurement  +  derivative low-pass filter
 *    D = -Kd * d(measured)/dt eliminates the derivative kick that
 *    occurs every time the setpoint steps (direction change, ramp
 *    increment).  A secondary EMA (α = 0.3) further smooths high-
 *    frequency encoder noise in the D term.
 *
 * 3. Clamping anti-windup
 *    Integration is paused when the output is at its limit AND the
 *    integrator would push it further into saturation.  This stops
 *    integral wind-up during startup, saturation, and direction
 *    transitions without needing a conditional reset.
 *
 * 4. Tighter PID limits  (output ±15, integral = output_limit / Ki)
 *    The feedforward provides the bulk of the PWM command; the PID
 *    only corrects the residual error (~10-20% of speed).  Capping
 *    the PID output at ±15 and sizing the integral limit so it alone
 *    can never saturate the output prevents authority fight between
 *    FF and PID.
 *
 * 5. Full PID state reset on direction reversal
 *    When a wheel direction changes: integral, setpoint, derivative
 *    state (last_measured + filtered_deriv), speed estimate, and
 *    slewed PWM are all zeroed.  This:
 *      a) Prevents derivative spikes from the sign flip.
 *      b) Ensures the setpoint ramp always starts from 0.
 *      c) Prevents integral wind-up from the opposite direction.
 *      d) Stops the slewed PWM carrying momentum into the new direction.
 *
 * 6. PWM slew-rate limiter  (300 PWM/s)
 *    The final PWM command is limited to change by at most
 *    300 PWM-units/s.  Running at ~1 kHz this means the output can
 *    change 0.3 PWM per ms → 0 to 60 in 200 ms.  This absorbs the
 *    FF dead-zone jump at startup and prevents any sudden motor jolt.
 *
 * ── KNOWN LIMITATION ──────────────────────────────────────────────
 *    Single-channel encoders cannot detect actual rotation direction.
 *    Direction is inferred from the commanded motor direction pin.
 *    If the motor is still coasting in the old direction when the pin
 *    changes, the speed estimate will be wrong for ~1 mechanical time
 *    constant (τ ≈ 0.2 s).  The full state reset + ramp + slew rate
 *    limiter mitigates this in practice.  A hardware fix (quadrature
 *    encoder) would eliminate it entirely.
 *
 * ── SERIAL PROTOCOL  (115200 baud, LF-terminated) ─────────────────
 *  Host → ESP32
 *    PING              → READY
 *    CMD_VEL:v,w       → (silent – low latency)
 *    STOP              → STOPPED
 *    STREAM_ON         → STREAM_ON
 *    STREAM_OFF        → STREAM_OFF
 *    RESET_ODOM        → ODOM_RESET
 *    STATUS            → STATUS,ctrl,stream,Lkp,Lki,Lkd,Rkp,Rki,Rkd
 *    TUNE:kp,ki,kd     → PID both …
 *    TUNEL:kp,ki,kd    → PID left …
 *    TUNER:kp,ki,kd    → PID right …
 *
 *  ESP32 → Host  (20 Hz when streaming)
 *    ODOM,t_ms,l_spd,r_spd,l_pwm,r_pwm,l_sp,r_sp,v,w,x,y,yaw
 *    WATCHDOG          (CMD_VEL timeout)
 */

#include <Arduino.h>
#include <math.h>

// ─── Pin map ──────────────────────────────────────────────────────────────────
#define LEFT_PWM      25
#define LEFT_DIR      26
#define LEFT_SC       34   // speed-counter (single-channel encoder)

#define RIGHT_PWM     27
#define RIGHT_DIR     33
#define RIGHT_SC      35

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

// ─── Velocity limits  (right-motor limited, no-load) ─────────────────────────
#define V_WHEEL_MAX  0.65f
#define V_MAX        V_WHEEL_MAX
#define W_MAX        (V_WHEEL_MAX / HALF_TRACK)   // ≈ 2.49 rad/s

// ─── Control tuning constants ─────────────────────────────────────────────────
// Setpoint ramp: 0.6 m/s² → 0 to 0.2 m/s in ~333 ms.
#define MAX_ACCEL_MS2       0.6f

// PWM slew: 300 PWM/s at 1 kHz → 0.3 PWM/ms → 0 to 60 in 200 ms.
#define PWM_SLEW_RATE       300.0f

// Dead zone minimum PWM: the minimum PWM that produces measurable wheel motion.
// Below this threshold the motor stalls.  When motion is commanded, the output
// is clamped to this minimum so the PID can never drive the motor into its dead
// zone and cause limit cycle oscillation (move-stop-move-stop).
// Values from the identified motor models (v = 0 when PWM ≤ dead zone boundary):
//   Left:  v = 0.0402*PWM - 0.3415 = 0  →  PWM = 8.5  →  use 9
//   Right: v = 0.0127*PWM + 0.0059 = 0  →  PWM < 0    →  no dead zone, use 2
#define LEFT_MIN_PWM   9
#define RIGHT_MIN_PWM  2

// Speed EMA: α=0.5 with the 200 ms window (raw estimates are already
// smoother; less aggressive filtering still gives τ ≈ 200 ms).
#define SPEED_FILTER_ALPHA  0.5f

// Derivative EMA: α=0.3 → stronger noise suppression on the D term.
#define DERIV_FILTER_ALPHA  0.3f

// ─── Timing ───────────────────────────────────────────────────────────────────
// SPEED_CALC_US is intentionally long (200 ms) to accumulate enough encoder
// pulses for accurate speed estimation at low speeds.
//
// At 0.09 m/s (spin case): 0.09/0.00576 * 0.05 = 0.8 pulses per 50 ms → the
// measurement alternates between 0 and 0.115 m/s → PID oscillates (jerk).
// At 200 ms:               0.09/0.00576 * 0.20 = 3.1 pulses → stable reading.
//
// The setpoint ramp runs on a SEPARATE 50 ms timer so it is not slowed down.
#define SPEED_CALC_US    200000UL  // 200 ms → 5 Hz speed update
#define RAMP_UPDATE_US    50000UL  // 50 ms  → 20 Hz setpoint ramp
#define ODOM_STREAM_MS   50UL      // 50 ms  → 20 Hz telemetry
#define CMD_TIMEOUT_MS   500UL     // watchdog
#define CTRL_MIN_US      1000UL    // control loop floor: 1 kHz

// ─── Encoder debounce ────────────────────────────────────────────────────────
#define DEBOUNCE_US  2000UL

// ─── Direction polarity ───────────────────────────────────────────────────────
#define LEFT_FORWARD   HIGH
#define RIGHT_FORWARD  LOW

// ─── PID structure ────────────────────────────────────────────────────────────
struct PIDController {
  float Kp, Ki, Kd;
  float setpoint;
  float integral;
  float last_measured;     // previous measurement for derivative-on-measurement
  float filtered_deriv;    // low-pass filtered derivative
  float output;
  float integral_limit;    // = output_limit / Ki  → I alone never saturates output
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

float leftSpeed  = 0.0f;   // EMA-filtered wheel speed (m/s, signed)
float rightSpeed = 0.0f;
int   leftPWM    = 0;      // last applied PWM (for telemetry only)
int   rightPWM   = 0;

unsigned long lastSpeedCalcUS = 0;
long lastLeftPulses  = 0;
long lastRightPulses = 0;

// Dead-reckoning odometry
float odomX   = 0.0f;
float odomY   = 0.0f;
float odomYaw = 0.0f;
float odomV   = 0.0f;
float odomW   = 0.0f;

PIDController leftPID;
PIDController rightPID;

// Setpoint targets: PID setpoints ramp toward these each speed cycle
float leftTargetSP  = 0.0f;
float rightTargetSP = 0.0f;

// Slewed PWM outputs (float for sub-step precision in slew limiter)
float leftPWMSlewed  = 0.0f;
float rightPWMSlewed = 0.0f;

bool          streaming              = false;
bool          speedCtrlEnabled       = false;
bool          speedUpdatedThisCycle  = false;  // set by updateSpeeds(), consumed by updateSpeedControl()
unsigned long lastStreamMS           = 0;
unsigned long lastCmdVelMS           = 0;
unsigned long lastCtrlUS             = 0;
unsigned long lastRampUS             = 0;

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
  pid.Kp             = kp;
  pid.Ki             = ki;
  pid.Kd             = kd;
  pid.setpoint       = 0.0f;
  pid.integral       = 0.0f;
  pid.last_measured  = 0.0f;
  pid.filtered_deriv = 0.0f;
  pid.output         = 0.0f;
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

  // ── Proportional ────────────────────────────────────────────────────────
  float P = pid.Kp * error;

  // ── Derivative on measurement  +  low-pass filter ───────────────────────
  // Using -d(measured)/dt instead of d(error)/dt means setpoint steps
  // (ramp increments, direction changes, new CMD_VEL) do NOT produce a
  // derivative spike.  Only actual changes in wheel speed affect D.
  // The EMA further suppresses encoder quantisation noise.
  float raw_deriv = -(measured - pid.last_measured) / dt;
  pid.filtered_deriv = DERIV_FILTER_ALPHA * raw_deriv
                     + (1.0f - DERIV_FILTER_ALPHA) * pid.filtered_deriv;
  float D = pid.Kd * pid.filtered_deriv;
  pid.last_measured = measured;

  // ── Clamping anti-windup ─────────────────────────────────────────────────
  // Do not integrate when the output is already at its limit AND the
  // integrator would push it further into saturation.  This prevents
  // runaway wind-up during startup, large disturbances, and direction
  // transitions without needing explicit manual resets everywhere.
  float I          = pid.Ki * pid.integral;
  float output_est = P + I + D;
  bool pos_sat = (output_est >=  pid.output_limit);
  bool neg_sat = (output_est <= -pid.output_limit);
  bool windup  = (pos_sat && error > 0.0f) || (neg_sat && error < 0.0f);
  if (!windup) {
    pid.integral = constrain(pid.integral + error * dt,
                             -pid.integral_limit, pid.integral_limit);
  }

  pid.output = constrain(P + pid.Ki * pid.integral + D,
                         -pid.output_limit, pid.output_limit);
  return pid.output;
}

// ─── Feedforward (inverted motor model) ───────────────────────────────────────
// fabsf(v): direction is handled by the direction pin; FF computes the
// PWM magnitude only.  Threshold 0.05 m/s keeps the motor off below the
// model's dead-zone (~PWM 8-9 for the left motor).
//
// RIGHT_FF_LOAD_FACTOR compensates for the right motor running slower under
// the robot's weight than the no-load model predicts.  The no-load model
// (K=0.0127) underestimates the PWM needed under load because the right
// motor has lower torque (3x lower gain than left).  Start at 1.25 and
// tune upward until the robot drives straight at v=0.2, w=0.  You can
// adjust this live via  TUNER:kp,ki,kd  or re-identify the model under load.
#define RIGHT_FF_LOAD_FACTOR  1.25f

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

void stopMotors() {
  speedCtrlEnabled       = false;
  leftTargetSP           = 0.0f;
  rightTargetSP          = 0.0f;
  leftPID.setpoint       = 0.0f;
  rightPID.setpoint      = 0.0f;
  leftPID.integral       = 0.0f;
  rightPID.integral      = 0.0f;
  leftPWMSlewed          = 0.0f;
  rightPWMSlewed         = 0.0f;
  applyLeftPWM(0);
  applyRightPWM(0);
  setLeftDir(true);
  setRightDir(true);
}

// ─── Differential drive ───────────────────────────────────────────────────────

/**
 * Uniformly scale (v, w) so neither wheel exceeds V_WHEEL_MAX.
 * The v/w ratio is preserved so the robot still curves the same way.
 */
void scaleCmdVel(float v, float w, float& vs, float& ws) {
  float vL   = v - w * HALF_TRACK;
  float vR   = v + w * HALF_TRACK;
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

  // ── Full PID state reset on direction reversal ────────────────────────────
  // Each field is zeroed for a specific reason:
  //   integral       – no wind-up from the previous direction carries over
  //   setpoint       – ramp always starts cleanly from 0 in the new direction
  //   last_measured  – prevents a large spike in the first derivative update
  //   filtered_deriv – clears the derivative filter memory
  //   speed estimate – single-channel encoder cannot verify actual direction;
  //                    zeroing avoids sign confusion until the motor settles
  //   slewed PWM     – stops the slew limiter carrying momentum across the flip
  if ((vL >= 0.0f) != leftFwd) {
    leftPID.integral       = 0.0f;
    leftPID.setpoint       = 0.0f;
    leftPID.last_measured  = 0.0f;
    leftPID.filtered_deriv = 0.0f;
    leftSpeed              = 0.0f;
    leftPWMSlewed          = 0.0f;
  }
  if ((vR >= 0.0f) != rightFwd) {
    rightPID.integral       = 0.0f;
    rightPID.setpoint       = 0.0f;
    rightPID.last_measured  = 0.0f;
    rightPID.filtered_deriv = 0.0f;
    rightSpeed              = 0.0f;
    rightPWMSlewed          = 0.0f;
  }

  setLeftDir(vL  >= 0.0f);
  setRightDir(vR >= 0.0f);

  leftTargetSP  = vL;
  rightTargetSP = vR;

  speedCtrlEnabled = true;
  lastCmdVelMS     = millis();
}

// ─── Speed + odometry update  (20 Hz) ─────────────────────────────────────────
void updateSpeeds() {
  unsigned long now = micros();
  if (now - lastSpeedCalcUS < SPEED_CALC_US) return;

  noInterrupts();
  long lp = leftPulses;
  long rp = rightPulses;
  interrupts();

  float dt = (now - lastSpeedCalcUS) * 1e-6f;

  // Raw speed from encoder pulse count over the 200 ms window.
  float rawL = (lp - lastLeftPulses)  * METERS_PER_PULSE / dt;
  float rawR = (rp - lastRightPulses) * METERS_PER_PULSE / dt;

  // Outlier rejection: GPIO 34/35 have no internal pull-ups, so PWM
  // switching noise can inject spurious pulses.  If the raw speed jumps
  // by more than MAX_SPEED_JUMP in one window it is almost certainly noise
  // (a real motor cannot accelerate that fast in 200 ms).  Discard it and
  // hold the previous filtered value instead.
  // MAX_SPEED_JUMP = 0.35 m/s per 200 ms = 1.75 m/s^2 (>> MAX_ACCEL_MS2).
  const float MAX_SPEED_JUMP = 0.35f;
  if (fabsf(rawL - leftSpeed)  > MAX_SPEED_JUMP) rawL = leftSpeed;
  if (fabsf(rawR - rightSpeed) > MAX_SPEED_JUMP) rawR = rightSpeed;

  // Exponential moving average: α=0.5 → new sample weight 50%, history 50%.
  leftSpeed  = SPEED_FILTER_ALPHA * rawL + (1.0f - SPEED_FILTER_ALPHA) * leftSpeed;
  rightSpeed = SPEED_FILTER_ALPHA * rawR + (1.0f - SPEED_FILTER_ALPHA) * rightSpeed;

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
  speedUpdatedThisCycle = true;   // signal PID to use fresh measurement
  // Setpoint ramp runs in updateRamp() on its own 50 ms timer.
}

// ─── Setpoint ramp  (20 Hz, independent of speed estimation) ─────────────────
// Running the ramp faster than the speed window ensures the reference
// velocity advances smoothly even though measured speed only updates at 5 Hz.
void updateRamp() {
  unsigned long now = micros();
  if (now - lastRampUS < RAMP_UPDATE_US) return;
  float dt = (now - lastRampUS) * 1e-6f;
  lastRampUS = now;

  const float max_step = MAX_ACCEL_MS2 * dt;
  leftPID.setpoint  += constrain(leftTargetSP  - leftPID.setpoint,  -max_step, max_step);
  rightPID.setpoint += constrain(rightTargetSP - rightPID.setpoint, -max_step, max_step);
}

// ─── Speed control (tied to speed update, 5 Hz) ───────────────────────────────
// The PID is driven only when a fresh speed measurement is available.

void updateSpeedControl() {
  if (!speedCtrlEnabled) return;

  if (millis() - lastCmdVelMS > CMD_TIMEOUT_MS) {
    stopMotors();
    Serial.println("WATCHDOG");
    return;
  }

  unsigned long now = micros();
  float dt = (now - lastCtrlUS) * 1e-6f;
  if (dt < 0.001f) return;   // still cap output loop at 1 kHz for slew
  lastCtrlUS = now;

  // PID: only recompute when a fresh speed measurement is available.
  // With a 200 ms speed window, calling pidUpdate() 200 times with the
  // same stale measured value accumulates integral 200x and produces a
  // derivative of 0 for 199 calls then a spike on the 200th.
  // The slew limiter below still runs at 1 kHz to keep the output smooth.
  if (speedUpdatedThisCycle) {
    speedUpdatedThisCycle = false;

    float l_pid = pidUpdate(leftPID,  leftSpeed);
    float r_pid = pidUpdate(rightPID, rightSpeed);

    if (leftPID.setpoint  < 0.0f) l_pid = -l_pid;
    if (rightPID.setpoint < 0.0f) r_pid = -r_pid;

    // Update slew targets with new PID output
    leftPWMSlewed  = leftFF(leftPID.setpoint)   + l_pid;
    rightPWMSlewed = rightFF(rightPID.setpoint) + r_pid;
  }

  // PWM slew-rate limiter runs every call (1 kHz) — smoothly advances the
  // actual output toward the last computed target between PID updates.
  float max_slew = PWM_SLEW_RATE * dt;
  float l_out = constrain((float)leftPWM  + constrain(leftPWMSlewed  - (float)leftPWM,  -max_slew, max_slew), 0.0f, (float)MAX_PWM);
  float r_out = constrain((float)rightPWM + constrain(rightPWMSlewed - (float)rightPWM, -max_slew, max_slew), 0.0f, (float)MAX_PWM);

  // Synchronized startup: prevent one motor from pulling ahead of the other
  // during the ramp-up phase.  Without this, the motor with the lower target
  // PWM (typically left, at 13.5 vs right at 15.3+) reaches its operating
  // speed first and the robot yaws before both motors are running.
  // Clamp each motor's output to the same FRACTION of its target as the
  // motor that is farthest behind.  Once both are within 1 PWM of their
  // targets the sync is released and they track independently.
  if (leftPWMSlewed > 1.0f && rightPWMSlewed > 1.0f) {
    float l_frac = l_out / leftPWMSlewed;
    float r_frac = r_out / rightPWMSlewed;
    float min_frac = min(l_frac, r_frac);
    // Only synchronize while still ramping up (below 95% of target)
    if (min_frac < 0.95f) {
      l_out = min(l_out, leftPWMSlewed  * min_frac + max_slew);
      r_out = min(r_out, rightPWMSlewed * min_frac + max_slew);
    }
  }

  // Dead zone clamping: when motion is commanded, never let the output fall
  // below the dead zone threshold.  Without this, any PID over-correction
  // stalls the motor; error then builds up and the motor surges back on,
  // creating the limit cycle (jerk) at low speeds.
  // Only applied when |setpoint| > 0.05 (i.e., motion is actually requested).
  if (fabsf(leftPID.setpoint)  > 0.05f) l_out = max(l_out, (float)LEFT_MIN_PWM);
  if (fabsf(rightPID.setpoint) > 0.05f) r_out = max(r_out, (float)RIGHT_MIN_PWM);

  applyLeftPWM( (int)roundf(l_out));
  applyRightPWM((int)roundf(r_out));
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

// Integral limit sized so the integrator alone cannot saturate the output.
static inline float integralLimit(float ki, float olim) {
  return (ki > 0.01f) ? (olim / ki) : 10.0f;
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
    }

  } else if (cmd.startsWith("TUNE:")) {
    float kp, ki, kd;
    if (parsePID(cmd.substring(5), kp, ki, kd)) {
      const float olim = 15.0f;
      pidInit(leftPID,  kp, ki, kd, integralLimit(ki, olim), olim);
      pidInit(rightPID, kp, ki, kd, integralLimit(ki, olim), olim);
      Serial.print("PID both Kp="); Serial.print(kp, 2);
      Serial.print(" Ki=");          Serial.print(ki, 2);
      Serial.print(" Kd=");          Serial.println(kd, 2);
    }

  } else if (cmd.startsWith("TUNEL:")) {
    float kp, ki, kd;
    if (parsePID(cmd.substring(6), kp, ki, kd)) {
      const float olim = 15.0f;
      pidInit(leftPID, kp, ki, kd, integralLimit(ki, olim), olim);
      Serial.print("PID left Kp="); Serial.print(kp, 2);
      Serial.print(" Ki=");          Serial.print(ki, 2);
      Serial.print(" Kd=");          Serial.println(kd, 2);
    }

  } else if (cmd.startsWith("TUNER:")) {
    float kp, ki, kd;
    if (parsePID(cmd.substring(6), kp, ki, kd)) {
      const float olim = 15.0f;
      pidInit(rightPID, kp, ki, kd, integralLimit(ki, olim), olim);
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

  // PID gains  –  output limit ±15, integral limit = output_limit / Ki
  // Left  Ki=2: integral_limit = 7.5  → I contribution capped at ±15 PWM
  // Right Ki=5: integral_limit = 3.0  → I contribution capped at ±15 PWM
  //
  // Right motor needs higher Ki because its gain (K=0.0127) is much lower
  // than left (K=0.0402), making it more sensitive to model error and
  // requiring stronger integral action for the same steady-state accuracy.
  //
  // If the robot still curves right after flashing, re-identify the motor
  // models under load – no-load models may under-predict right motor speed.
  pidInit(leftPID,  10.0f, 2.0f, 0.1f, 7.5f, 15.0f);
  pidInit(rightPID, 10.0f, 5.0f, 0.1f, 3.0f, 15.0f);

  setLeftDir(true);
  setRightDir(true);
  applyLeftPWM(0);
  applyRightPWM(0);

  lastSpeedCalcUS = micros();
  lastCtrlUS      = micros();
  lastRampUS      = micros();
  lastCmdVelMS    = millis();

  Serial.println("READY");
  Serial.print("LIMITS,V_MAX="); Serial.print(V_MAX,  3);
  Serial.print(",W_MAX=");        Serial.print(W_MAX,  3);
  Serial.print(",L=");            Serial.println(WHEEL_SEPARATION_M, 3);
}

void loop() {
  updateSpeeds();         // 5 Hz  – speed estimation (200 ms window)
  updateRamp();           // 20 Hz – setpoint advance
  updateSpeedControl();   // 1 kHz – PID + slew + PWM
  streamOdom();           // 20 Hz – telemetry

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() > 0) {
      handleCommand(cmd);
    }
  }
}
