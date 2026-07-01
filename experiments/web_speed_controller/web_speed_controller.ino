#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <cmath>

// ── WiFi credentials ──────────────────────────────────────────────
const char* ssid = "Airtel 4G MiFi_80C1";
const char* password = "55105322Gh$";

// ── Pin definitions ──────────────────────────────────────────────
#define LEFT_PWM     25
#define LEFT_DIR     26
#define LEFT_SC      34

#define RIGHT_PWM    27
#define RIGHT_DIR    33
#define RIGHT_SC     35

// ── LEDC channels ────────────────────────────────────────────────
#define LEFT_PWM_CH   2
#define RIGHT_PWM_CH  1

// ── ZS-X11D1 Compatible PWM Settings ────────────────────────────
#define PWM_FREQ      20000
#define PWM_RES       10              // 10-bit resolution (0-1023)
#define MAX_DUTY      80              // Maximum duty cycle in percent
#define MAX_PWM_VALUE (int)((MAX_DUTY / 100.0) * ((1 << PWM_RES) - 1))  // 0.8 * 1023 = 818

// ── Robot parameters ──────────────────────────────────────────────
#define WHEEL_DIAMETER 0.165
#define WHEEL_BASE 0.52
#define PULSES_PER_REV 90
#define WHEEL_CIRCUMFERENCE (WHEEL_DIAMETER * 3.14159)
#define MAX_SPEED 1.5

// ── SAFETY: Soft-Start and Jerk Limiting ────────────────────────
#define SOFT_START_RATE 0.15
#define JERK_LIMIT 0.3
#define INITIAL_PWM_LIMIT 0.3
#define SMOOTHING_FACTOR 0.7

// ── Encoder noise filtering ──────────────────────────────────────
#define ENCODER_DEBOUNCE_US 1000
#define MIN_PULSES_FOR_SPEED 2

// ── PID Speed Control Parameters ─────────────────────────────────
float Kp = 0.6;
float Ki = 0.05;
float Kd = 0.03;

// ── Motor compensation ───────────────────────────────────────────
#define RIGHT_COMPENSATION 2.26

// ── Encoder counts ───────────────────────────────────────────────
volatile long leftPulses = 0;
volatile long rightPulses = 0;
volatile unsigned long lastLeftPulseTime = 0;
volatile unsigned long lastRightPulseTime = 0;

// ── Speed Control Variables ──────────────────────────────────────
float targetLeftSpeed = 0;
float targetRightSpeed = 0;
float currentLeftSpeed = 0;
float currentRightSpeed = 0;

float smoothLeftSpeed = 0;
float smoothRightSpeed = 0;
float commandedLeftSpeed = 0;
float commandedRightSpeed = 0;
float previousLeftSpeed = 0;
float previousRightSpeed = 0;

float leftPIDOutput = 0;
float rightPIDOutput = 0;
float leftIntegral = 0;
float rightIntegral = 0;
float leftPrevError = 0;
float rightPrevError = 0;

bool motorsRunning = false;
bool emergencyStop = false;
unsigned long startTime = 0;
const unsigned long SAFETY_DELAY = 500;

unsigned long lastPIDTime = 0;
const unsigned long PID_INTERVAL = 30;

// ── Speed Lookup Table (Now using duty cycle percentage) ────────
struct SpeedMap {
  float duty_percent;    // Duty cycle in percent (0-80%)
  float speed_mps;
};

const SpeedMap speedTable[] = {
  {5.0, 0.0547},    // 5% duty
  {10.0, 0.3208},   // 10% duty
  {15.0, 0.2966},   // 15% duty
  {20.0, 0.4377},   // 20% duty
  {25.0, 0.5777},   // 25% duty
  {30.0, 0.7009},   // 30% duty
  {35.0, 0.8311},   // 35% duty
  {40.0, 0.9688},   // 40% duty
  {45.0, 1.1300},   // 45% duty
  {50.0, 1.2913},   // 50% duty
  {55.0, 1.4480},   // 55% duty
  {60.0, 1.5683},   // 60% duty
  {65.0, 1.6346},   // 65% duty
  {70.0, 1.6864},   // 70% duty
  {75.0, 1.7313},   // 75% duty
  {80.0, 1.7734},   // 80% duty
};

const int TABLE_SIZE = 16;

// ── Web server ──────────────────────────────────────────────────
WebServer server(80);

// ── Helper: Convert duty percentage to PWM value ────────────────
int dutyToPWM(float dutyPercent) {
  dutyPercent = constrain(dutyPercent, 0, MAX_DUTY);
  return (int)((dutyPercent / 100.0) * ((1 << PWM_RES) - 1));
}

// ── Helper: Convert PWM value to duty percentage ────────────────
float pwmToDuty(int pwmValue) {
  pwmValue = constrain(pwmValue, 0, ((1 << PWM_RES) - 1));
  return (pwmValue / (float)((1 << PWM_RES) - 1)) * 100.0;
}

// ── ISRs ─────────────────────────────────────────────────────────
void IRAM_ATTR leftISR() {
  unsigned long now = micros();
  if (now - lastLeftPulseTime > ENCODER_DEBOUNCE_US) {
    leftPulses++;
    lastLeftPulseTime = now;
  }
}

void IRAM_ATTR rightISR() {
  unsigned long now = micros();
  if (now - lastRightPulseTime > ENCODER_DEBOUNCE_US) {
    rightPulses++;
    lastRightPulseTime = now;
  }
}

// ── Lookup Functions (Using Duty Cycle) ─────────────────────────
int getPWMForSpeed(float targetSpeed) {
  if (targetSpeed <= 0) return 0;
  
  float duty = 0;
  if (targetSpeed <= speedTable[0].speed_mps) {
    duty = speedTable[0].duty_percent;
  } else if (targetSpeed >= speedTable[TABLE_SIZE-1].speed_mps) {
    duty = speedTable[TABLE_SIZE-1].duty_percent;
  } else {
    for (int i = 0; i < TABLE_SIZE-1; i++) {
      if (targetSpeed >= speedTable[i].speed_mps && targetSpeed <= speedTable[i+1].speed_mps) {
        float fraction = (targetSpeed - speedTable[i].speed_mps) / 
                         (speedTable[i+1].speed_mps - speedTable[i].speed_mps);
        duty = speedTable[i].duty_percent + fraction * (speedTable[i+1].duty_percent - speedTable[i].duty_percent);
        break;
      }
    }
  }
  
  return dutyToPWM(duty);
}

// ── SAFE Motor Control ──────────────────────────────────────────
void setMotorPWM(int leftPWM, int rightPWM) {
  static float leftPWM_soft = 0;
  static float rightPWM_soft = 0;
  
  // Clamp to MAX_PWM_VALUE (not MAX_PWM anymore)
  leftPWM = constrain(leftPWM, 0, MAX_PWM_VALUE);
  rightPWM = constrain(rightPWM, 0, MAX_PWM_VALUE);
  
  // Gradual PWM change (soft-start)
  float leftDelta = leftPWM - leftPWM_soft;
  float rightDelta = rightPWM - rightPWM_soft;
  
  float maxChange = MAX_PWM_VALUE * 0.05;
  leftDelta = constrain(leftDelta, -maxChange, maxChange);
  rightDelta = constrain(rightDelta, -maxChange, maxChange);
  
  leftPWM_soft += leftDelta;
  rightPWM_soft += rightDelta;
  
  leftPWM = constrain((int)leftPWM_soft, 0, MAX_PWM_VALUE);
  rightPWM = constrain((int)rightPWM_soft, 0, MAX_PWM_VALUE);
  
  if (emergencyStop) {
    leftPWM = 0;
    rightPWM = 0;
  }
  
  if (leftPWM > 0 || rightPWM > 0) {
    motorsRunning = true;
  } else {
    motorsRunning = false;
  }
  
  digitalWrite(LEFT_DIR, HIGH);
  digitalWrite(RIGHT_DIR, LOW);
  
  ledcWrite(LEFT_PWM_CH, leftPWM);
  ledcWrite(RIGHT_PWM_CH, rightPWM);
}

void emergencyStopMotors() {
  emergencyStop = true;
  ledcWrite(LEFT_PWM_CH, 0);
  ledcWrite(RIGHT_PWM_CH, 0);
  targetLeftSpeed = 0;
  targetRightSpeed = 0;
  motorsRunning = false;
  leftIntegral = 0;
  rightIntegral = 0;
  leftPrevError = 0;
  rightPrevError = 0;
  Serial.println("⚠️ EMERGENCY STOP ACTIVATED!");
}

void stopMotors() {
  emergencyStop = false;
  ledcWrite(LEFT_PWM_CH, 0);
  ledcWrite(RIGHT_PWM_CH, 0);
  targetLeftSpeed = 0;
  targetRightSpeed = 0;
  smoothLeftSpeed = 0;
  smoothRightSpeed = 0;
  commandedLeftSpeed = 0;
  commandedRightSpeed = 0;
  motorsRunning = false;
  leftIntegral = 0;
  rightIntegral = 0;
  leftPrevError = 0;
  rightPrevError = 0;
}

// ── Differential Drive Kinematics ───────────────────────────────
void calculateWheelSpeeds(float linearSpeed, float angularSpeed) {
  float leftSpeed = linearSpeed - (angularSpeed * WHEEL_BASE) / 2.0;
  float rightSpeed = linearSpeed + (angularSpeed * WHEEL_BASE) / 2.0;
  
  leftSpeed = constrain(leftSpeed, -MAX_SPEED, MAX_SPEED);
  rightSpeed = constrain(rightSpeed, -MAX_SPEED, MAX_SPEED);
  
  targetLeftSpeed = leftSpeed;
  targetRightSpeed = rightSpeed;
}

// ── PID Speed Controller ─────────────────────────────────────────
float computePID(float target, float current, float &integral, float &prevError) {
  float error = target - current;
  
  if (motorsRunning && abs(error) < target * 0.5 && !emergencyStop) {
    integral += error * (PID_INTERVAL / 1000.0);
    integral = constrain(integral, -50, 50);
  } else {
    integral *= 0.95;
  }
  
  float derivative = (error - prevError) / (PID_INTERVAL / 1000.0);
  derivative = constrain(derivative, -20, 20);
  
  float output = (Kp * error) + (Ki * integral) + (Kd * derivative);
  output = constrain(output, -50, 50);
  
  prevError = error;
  return output;
}

// ── Update Speed Control ─────────────────────────────────────────
void updateSpeedControl() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastPIDTime >= PID_INTERVAL) {
    // Soft-start
    float speedDiffLeft = targetLeftSpeed - commandedLeftSpeed;
    float speedDiffRight = targetRightSpeed - commandedRightSpeed;
    
    float maxAccel = SOFT_START_RATE * (PID_INTERVAL / 1000.0);
    if (abs(speedDiffLeft) > maxAccel) {
      commandedLeftSpeed += sign(speedDiffLeft) * maxAccel;
    } else {
      commandedLeftSpeed = targetLeftSpeed;
    }
    
    if (abs(speedDiffRight) > maxAccel) {
      commandedRightSpeed += sign(speedDiffRight) * maxAccel;
    } else {
      commandedRightSpeed = targetRightSpeed;
    }
    
    // Jerk limiting
    float jerkLimit = JERK_LIMIT * (PID_INTERVAL / 1000.0);
    float leftJerk = commandedLeftSpeed - previousLeftSpeed;
    float rightJerk = commandedRightSpeed - previousRightSpeed;
    
    if (abs(leftJerk) > jerkLimit) {
      commandedLeftSpeed = previousLeftSpeed + sign(leftJerk) * jerkLimit;
    }
    if (abs(rightJerk) > jerkLimit) {
      commandedRightSpeed = previousRightSpeed + sign(rightJerk) * jerkLimit;
    }
    
    previousLeftSpeed = commandedLeftSpeed;
    previousRightSpeed = commandedRightSpeed;
    
    // Read encoders
    noInterrupts();
    long lCount = leftPulses;
    long rCount = rightPulses;
    leftPulses = 0;
    rightPulses = 0;
    interrupts();
    
    float dt = (currentTime - lastPIDTime) / 1000.0;
    
    // Calculate speeds
    if (motorsRunning && lCount >= MIN_PULSES_FOR_SPEED) {
      currentLeftSpeed = (lCount / (float)PULSES_PER_REV) * WHEEL_CIRCUMFERENCE / dt;
    } else if (!motorsRunning) {
      currentLeftSpeed = 0;
    } else {
      currentLeftSpeed *= 0.9;
    }
    
    if (motorsRunning && rCount >= MIN_PULSES_FOR_SPEED) {
      currentRightSpeed = (rCount / (float)PULSES_PER_REV) * WHEEL_CIRCUMFERENCE / dt;
    } else if (!motorsRunning) {
      currentRightSpeed = 0;
    } else {
      currentRightSpeed *= 0.9;
    }
    
    if (currentLeftSpeed > MAX_SPEED * 1.5) currentLeftSpeed = 0;
    if (currentRightSpeed > MAX_SPEED * 1.5) currentRightSpeed = 0;
    
    // Get PWM from lookup
    int leftBasePWM = getPWMForSpeed(commandedLeftSpeed);
    int rightBasePWM = getPWMForSpeed(commandedRightSpeed);
    
    // Apply compensation
    rightBasePWM = rightBasePWM * RIGHT_COMPENSATION;
    rightBasePWM = constrain(rightBasePWM, 0, MAX_PWM_VALUE);
    
    // PID correction
    if (motorsRunning && commandedLeftSpeed > 0.01 && !emergencyStop) {
      leftPIDOutput = computePID(commandedLeftSpeed, currentLeftSpeed, leftIntegral, leftPrevError);
    } else {
      leftPIDOutput = 0;
      leftIntegral = 0;
    }
    
    if (motorsRunning && commandedRightSpeed > 0.01 && !emergencyStop) {
      rightPIDOutput = computePID(commandedRightSpeed, currentRightSpeed, rightIntegral, rightPrevError);
    } else {
      rightPIDOutput = 0;
      rightIntegral = 0;
    }
    
    int leftPWM = leftBasePWM + leftPIDOutput;
    int rightPWM = rightBasePWM + rightPIDOutput;
    
    // Initial PWM limit
    if (millis() - startTime < SAFETY_DELAY) {
      leftPWM = leftPWM * INITIAL_PWM_LIMIT;
      rightPWM = rightPWM * INITIAL_PWM_LIMIT;
    }
    
    setMotorPWM(leftPWM, rightPWM);
    
    lastPIDTime = currentTime;
  }
}

// ── Helper functions ─────────────────────────────────────────────
float sign(float x) {
  return (x > 0) - (x < 0);
}

// ── Web Handlers ──────────────────────────────────────────────────
void handleRoot() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <title>Safe Robot Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; text-align: center; margin: 20px; }
        .container { max-width: 700px; margin: auto; }
        .control-group { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .feature-badge { display: inline-block; background: #4CAF50; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; margin: 2px; }
        .safety-badge { display: inline-block; background: #ff9800; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; margin: 2px; }
        input[type="range"] { width: 80%; margin: 10px; }
        button { padding: 12px 24px; margin: 5px; font-size: 16px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 5px; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        button.danger { background: #f44336; }
        button.emergency { background: #d32f2f; padding: 16px 32px; font-size: 20px; font-weight: bold; }
        #status { margin: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; }
        .telemetry { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0; }
        .telemetry-item { padding: 10px; background: #f5f5f5; border-radius: 5px; }
        .telemetry-item .label { font-weight: bold; color: #666; }
        .telemetry-item .value { font-size: 20px; color: #2196F3; }
        .info-box { background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Safe Robot Controller</h1>
        <div class="feature-badge">⚡ 20 kHz PWM</div>
        <div class="feature-badge">🔇 Silent</div>
        <div class="feature-badge">📊 10-bit (0-1023)</div>
        <div class="safety-badge">🚀 Soft-Start</div>
        <div class="safety-badge">⚠️ Jerk Limited</div>
        <div class="safety-badge">🛑 Anti-Windup</div>
        
        <div class="info-box">
            ⚙️ Max Duty Cycle: 80% | Max PWM Value: 818
        </div>
        
        <div class="control-group">
            <h3>Speed Control</h3>
            <div>
                <label>Linear Speed: <span id="speedValue">0.50</span> m/s</label><br>
                <input type="range" id="speedSlider" min="0" max="1.5" step="0.01" value="0.50">
            </div>
            <div>
                <label>Angular Velocity (ω): <span id="omegaValue">0.00</span> rad/s</label><br>
                <input type="range" id="omegaSlider" min="-2.0" max="2.0" step="0.01" value="0.0">
            </div>
            <br>
            <button onclick="startControl()" id="startBtn">▶ Start</button>
            <button onclick="stopControl()" class="danger">⏹ Stop</button>
            <br>
            <button onclick="emergencyStop()" class="emergency">🛑 EMERGENCY STOP</button>
        </div>
        
        <div id="status">Status: Ready</div>
        
        <div class="telemetry">
            <div class="telemetry-item">
                <div class="label">Left Speed</div>
                <div class="value" id="leftSpeed">0.00 m/s</div>
            </div>
            <div class="telemetry-item">
                <div class="label">Right Speed</div>
                <div class="value" id="rightSpeed">0.00 m/s</div>
            </div>
            <div class="telemetry-item">
                <div class="label">Commanded Left</div>
                <div class="value" id="cmdLeft">0.00 m/s</div>
            </div>
            <div class="telemetry-item">
                <div class="label">Commanded Right</div>
                <div class="value" id="cmdRight">0.00 m/s</div>
            </div>
        </div>
        
        <script>
            let running = false;
            
            document.getElementById('speedSlider').addEventListener('input', function() {
                document.getElementById('speedValue').textContent = parseFloat(this.value).toFixed(2);
            });
            
            document.getElementById('omegaSlider').addEventListener('input', function() {
                document.getElementById('omegaValue').textContent = parseFloat(this.value).toFixed(2);
            });
            
            function startControl() {
                if (running) return;
                const speed = document.getElementById('speedSlider').value;
                const omega = document.getElementById('omegaSlider').value;
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('status').textContent = 'Status: Soft-start...';
                running = true;
                
                fetch('/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: 'speed=' + speed + '&omega=' + omega
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        pollTelemetry();
                    }
                });
            }
            
            function stopControl() {
                fetch('/stop', { method: 'POST' })
                .then(() => {
                    document.getElementById('status').textContent = 'Status: Stopped';
                    document.getElementById('startBtn').disabled = false;
                    running = false;
                });
            }
            
            function emergencyStop() {
                if (confirm('⚠️ EMERGENCY STOP! Are you sure?')) {
                    fetch('/emergency', { method: 'POST' })
                    .then(() => {
                        document.getElementById('status').textContent = 'Status: ⚠️ EMERGENCY STOP';
                        document.getElementById('startBtn').disabled = false;
                        running = false;
                    });
                }
            }
            
            function pollTelemetry() {
                fetch('/telemetry')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('leftSpeed').textContent = data.leftSpeed.toFixed(3) + ' m/s';
                    document.getElementById('rightSpeed').textContent = data.rightSpeed.toFixed(3) + ' m/s';
                    document.getElementById('cmdLeft').textContent = data.cmdLeft.toFixed(3) + ' m/s';
                    document.getElementById('cmdRight').textContent = data.cmdRight.toFixed(3) + ' m/s';
                    
                    if (data.running) {
                        document.getElementById('status').textContent = 'Status: Running';
                        setTimeout(pollTelemetry, 100);
                    }
                });
            }
        </script>
    </div>
</body>
</html>
  )rawliteral";
  server.send(200, "text/html", html);
}

void handleStart() {
  if (server.hasArg("speed") && server.hasArg("omega")) {
    float speed = server.arg("speed").toFloat();
    float omega = server.arg("omega").toFloat();
    
    speed = constrain(speed, 0, MAX_SPEED);
    omega = constrain(omega, -2.0, 2.0);
    
    emergencyStop = false;
    startTime = millis();
    
    noInterrupts();
    leftPulses = 0;
    rightPulses = 0;
    interrupts();
    
    calculateWheelSpeeds(speed, omega);
    
    leftIntegral = 0;
    rightIntegral = 0;
    leftPrevError = 0;
    rightPrevError = 0;
    
    commandedLeftSpeed = 0;
    commandedRightSpeed = 0;
    previousLeftSpeed = 0;
    previousRightSpeed = 0;
    
    lastPIDTime = millis();
    motorsRunning = true;
    
    server.send(200, "application/json", 
                "{\"status\":\"success\",\"message\":\"Speed: " + String(speed) + 
                ", Omega: " + String(omega) + "\"}");
  } else {
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Missing parameters\"}");
  }
}

void handleStop() {
  stopMotors();
  server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Stopped\"}");
}

void handleEmergency() {
  emergencyStopMotors();
  server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Emergency stop activated\"}");
}

void handleTelemetry() {
  String json = "{";
  json += "\"running\":" + String(motorsRunning ? "true" : "false") + ",";
  json += "\"leftSpeed\":" + String(currentLeftSpeed, 3) + ",";
  json += "\"rightSpeed\":" + String(currentRightSpeed, 3) + ",";
  json += "\"cmdLeft\":" + String(commandedLeftSpeed, 3) + ",";
  json += "\"cmdRight\":" + String(commandedRightSpeed, 3);
  json += "}";
  server.send(200, "application/json", json);
}

// ── Setup ────────────────────────────────────────────────────────
void setup()
{
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n🛡️ Safe Motor Controller");
  Serial.println("=======================");
  Serial.printf("PWM Frequency: %d Hz\n", PWM_FREQ);
  Serial.printf("PWM Resolution: %d-bit (0-%d)\n", PWM_RES, (1 << PWM_RES) - 1);
  Serial.printf("Max Duty Cycle: %d%% (PWM Value: %d)\n", MAX_DUTY, MAX_PWM_VALUE);
  Serial.printf("Soft-start Rate: %.2f m/s²\n", SOFT_START_RATE);
  Serial.printf("Jerk Limit: %.2f m/s\n", JERK_LIMIT);
  Serial.println("✅ Safety features enabled");
  
  pinMode(LEFT_DIR,  OUTPUT);
  pinMode(RIGHT_DIR, OUTPUT);
  pinMode(LEFT_SC,  INPUT_PULLUP);
  pinMode(RIGHT_SC, INPUT_PULLUP);
  
  attachInterrupt(digitalPinToInterrupt(LEFT_SC),  leftISR,  RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_SC), rightISR, RISING);
  
  ledcSetup(LEFT_PWM_CH,  PWM_FREQ, PWM_RES);
  ledcSetup(RIGHT_PWM_CH, PWM_FREQ, PWM_RES);
  ledcAttachPin(LEFT_PWM,  LEFT_PWM_CH);
  ledcAttachPin(RIGHT_PWM, RIGHT_PWM_CH);
  
  ledcWrite(LEFT_PWM_CH,  0);
  ledcWrite(RIGHT_PWM_CH, 0);
  
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  
  if (MDNS.begin("esp32-robot")) {
    Serial.println("MDNS responder started");
    Serial.println("Access at http://esp32-robot.local");
  }
  
  server.on("/", handleRoot);
  server.on("/start", HTTP_POST, handleStart);
  server.on("/stop", HTTP_POST, handleStop);
  server.on("/emergency", HTTP_POST, handleEmergency);
  server.on("/telemetry", handleTelemetry);
  
  server.begin();
  Serial.println("HTTP server started");
  Serial.println("\n✅ Robot ready with safety features!");
}

// ── Loop ─────────────────────────────────────────────────────────
void loop()
{
  server.handleClient();
  
  if (motorsRunning && (targetLeftSpeed > 0.01 || targetRightSpeed > 0.01) && !emergencyStop) {
    updateSpeedControl();
  } else if (motorsRunning && emergencyStop) {
    ledcWrite(LEFT_PWM_CH, 0);
    ledcWrite(RIGHT_PWM_CH, 0);
  }
  
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint >= 1000) {
    if (motorsRunning && !emergencyStop) {
      Serial.print("Cmd: L=");
      Serial.print(commandedLeftSpeed, 2);
      Serial.print(" R=");
      Serial.print(commandedRightSpeed, 2);
      Serial.print(" | Actual: L=");
      Serial.print(currentLeftSpeed, 2);
      Serial.print(" R=");
      Serial.print(currentRightSpeed, 2);
      Serial.print(" | PWM: L=");
      Serial.print(ledcRead(LEFT_PWM_CH));
      Serial.print(" R=");
      Serial.print(ledcRead(RIGHT_PWM_CH));
      Serial.println();
    }
    lastPrint = millis();
  }
}