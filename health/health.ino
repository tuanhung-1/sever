#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <Wire.h>
#include "MAX30105.h"
#include <ClosedCube_MAX30205.h>
#include <MPU6050_tockn.h>
#include <ArduinoJson.h>
#include <unordered_map>
#include <string>
#include "secrets.h"

#define DELTA_G_THRESHOLD   0.9f
#define DELTA_G_WINDOW_MS   150
#define CONFIRM_WAIT_MS     250
#define AZ_FALLEN_THRESH    0.6f
#define AZ_NORMAL_RECOVERY  0.65f
#define PEAK_G_INSTANT      3.0f
#define FALL_COOLDOWN_MS    5000

#define FILTER_GPEAK_MIN      1.6f
#define FILTER_DTHETA_MIN_DEG 35.0f
#define THETA_PRE_WINDOW      25
#define THETA_POST_WINDOW     25
#define FILTER_VAR_G_MAX      0.8f
#define VAR_G_POST_WINDOW     80

#define FALL_MIN_PEAK_G          1.4f
#define FALL_MEDIUM_PEAK_G       1.7f
#define FALL_STRONG_PEAK_G       2.4f

#define FALL_THETA_SOFT          18.0f
#define FALL_THETA_STRONG        35.0f

#define FALL_STILL_RATIO         0.75f
#define FALL_VERY_STILL_RATIO    0.88f

#define FALL_GYRO_STILL          18.0f
#define FALL_GYRO_ROTATE          100.0f
#define FALL_GYRO_CHAIR          120.0f

#define FALL_SCORE_PASS         5
#define FALL_JERK_STILL          1.2f

#define SAMPLE_INTERVAL_MS  10
#define BATCH_SIZE          200
#define BUZZER_PIN          19
#define BUZZER_CMD_MS       60000
#define BASELINE_WINDOW     150

#define PRE_SAMPLES         100
#define POST_SAMPLES        70
#define LSTM_WINDOW_SIZE    (PRE_SAMPLES + POST_SAMPLES)

#define CB_SIZE             1200
#define JSON_BUF_SIZE       20000
#define NUM_CHANNELS        8

#define SCALE_ACC       1000
#define SCALE_GYRO      10
#define SCALE_MAG       1000
#define SCALE_JERK      1000
#define REASON_MAX_LEN  64
#define BINARY_BUF_SIZE 5000

#define PPG_SENSOR_SAMPLE_RATE_HZ 50
#define PPG_FIFO_SAMPLE_AVERAGE   1

#define SPO2_WINDOW_SIZE 400
#define SPO2_STEP_SIZE   200

#define SPO2_MIN_IR              50000
#define SPO2_MIN_RED             10000

#define SPO2_MIN_IR_P2P          500
#define SPO2_MIN_RED_P2P         250

#define SPO2_MIN_ACDC              0.003f
#define SPO2_MAX_ACDC              0.25f

#define SPO2_MAX_JERK_SAMPLE       20.0f
#define SPO2_MAX_BAD_MOTION_RATE   0.20f

#define SPO2_BAD_LIMIT 200

#define TEMP_EMA_ALPHA   0.07f
#define TEMP_SKIN_OFFSET 1.5f

static uint8_t binaryBuf[BINARY_BUF_SIZE];

TwoWire             I2C_MAX = TwoWire(1);
MAX30105            max30102;
ClosedCube_MAX30205 max30205;
MPU6050             mpu6050(Wire);
WiFiClientSecure    espClient;
PubSubClient        mqttClient(espClient);

struct FallSample {
  float ax, ay, az;
  float gx, gy, gz;
  float mag;
  float jerk;
  float theta;
  unsigned long ts;
};

struct PpgSample {
  uint32_t ir;
  uint32_t red;
  unsigned long ts;
  bool     valid;
};

FallSample fallBuf[CB_SIZE];
int fallWriteIdx = 0;
int fallReadIdx  = 0;

PpgSample ppgBuf[SPO2_WINDOW_SIZE];
int  ppgWriteIdx   = 0;
int  ppgCount      = 0;
int  ppgNewSamples = 0;
bool ppgGapFlag    = false;
uint32_t ppgSeqOut = 0;

float   baselineG     = 1.0f;
float   baselineSumG  = 0.0f;
int     baselineCount = 0;
float   prevMag       = 1.0f;
int     spo2BadCount  = 0;

typedef enum {
  STATE_STABLE,
  STATE_TRANSITION,
  STATE_CONFIRMING,
  STATE_POSTCAPTURE
} FallState;

FallState     currentState     = STATE_STABLE;
unsigned long stateEnteredMs   = 0;
unsigned long lastFallAlert    = 0;
char          triggerReason[REASON_MAX_LEN] = "";
float         transitionPeakG  = 0.0f;
float         transitionDeltaG = 0.0f;
unsigned long triggerTime      = 0;
int           postSampleCount  = 0;

float         peakGInWindow = 0.0f;
float         maxDeltaG     = 0.0f;
unsigned long windowStart   = 0;
int           peakWriteIdx  = 0;

float         temperature    = 36.5f;
bool          tempInitialized = false;
unsigned long lastTempRead   = 0;
unsigned long lastSample     = 0;
unsigned long lastLog        = 0;
bool          buzzerOn       = false;
unsigned long buzzerTimer    = 0;
int           buzzerDuration = 0;

uint32_t filterRejectCount = 0;
uint32_t filterPassCount   = 0;

bool          freeFallDetected = false;
unsigned long freeFallTime     = 0;

static char* jsonBuf = nullptr;

unsigned long ppgLastTs       = 0;
uint32_t      ppgSampleCount  = 0;
float         ppgMeasuredFs   = PPG_SENSOR_SAMPLE_RATE_HZ;
#define PPG_FS_MEASURE_WINDOW 100

inline int cbCount() {
  return (fallWriteIdx - fallReadIdx + CB_SIZE) % CB_SIZE;
}

void cbPush(const FallSample& s) {
  fallBuf[fallWriteIdx] = s;
  fallWriteIdx = (fallWriteIdx + 1) % CB_SIZE;
  if (fallWriteIdx == fallReadIdx)
    fallReadIdx = (fallReadIdx + 1) % CB_SIZE;
}

void buzzerBip(int duration_ms) {
  digitalWrite(BUZZER_PIN, HIGH);
  buzzerOn       = true;
  buzzerTimer    = millis();
  buzzerDuration = duration_ms;
}

void buzzerUpdate() {
  if (buzzerOn && millis() - buzzerTimer > (unsigned long)buzzerDuration) {
    digitalWrite(BUZZER_PIN, LOW);
    buzzerOn = false;
  }
}

inline uint8_t* writeU8(uint8_t* p, uint8_t v)    { *p = v; return p + 1; }
inline uint8_t* writeU16BE(uint8_t* p, uint16_t v) { p[0]=(v>>8)&0xFF; p[1]=v&0xFF; return p+2; }
inline uint8_t* writeI16BE(uint8_t* p, int16_t v)  { p[0]=(v>>8)&0xFF; p[1]=v&0xFF; return p+2; }
inline uint8_t* writeU32BE(uint8_t* p, uint32_t v) { p[0]=(v>>24)&0xFF;p[1]=(v>>16)&0xFF;p[2]=(v>>8)&0xFF;p[3]=v&0xFF; return p+4; }

inline int16_t floatToI16(float v, float scale) {
  float s = v * scale;
  if (s >  32767.0f) s =  32767.0f;
  if (s < -32768.0f) s = -32768.0f;
  return (int16_t)s;
}

inline float calcTheta(float ax, float ay, float az) {
  float lateral = sqrtf(ax * ax + ay * ay);
  return degrees(atan2f(lateral, fabsf(az)));
}

float median(float* arr, int n) {
  for (int i = 0; i < n-1; i++)
    for (int j = i+1; j < n; j++)
      if (arr[j] < arr[i]) { float t=arr[i]; arr[i]=arr[j]; arr[j]=t; }
  return arr[n/2];
}

struct PPGQuality {
  bool ok;
  char reason[64];
  float irMean;
  float redMean;
  float irP2P;
  float redP2P;
  float irACDC;
  float redACDC;
  float maxJerk;
  float badMotionRate;
  float measuredFs;
};
void resetPpgWindow();
void pushPpgSample(const PpgSample& p);
int spo2Index(int logicalIndex);
void updatePpgFs(unsigned long nowTs);
void readPpgStream();
bool checkPPGQuality(PPGQuality& q);
void sendSensorWindow();
void resetPpgWindow() {
  ppgWriteIdx   = 0;
  ppgCount      = 0;
  ppgNewSamples = 0;
  ppgGapFlag    = true;
  ppgLastTs     = 0;
  ppgSampleCount = 0;
}

void pushPpgSample(const PpgSample& p) {
  ppgBuf[ppgWriteIdx] = p;
  ppgWriteIdx = (ppgWriteIdx + 1) % SPO2_WINDOW_SIZE;
  if (ppgCount < SPO2_WINDOW_SIZE) ppgCount++;
  if (ppgNewSamples < SPO2_STEP_SIZE) ppgNewSamples++;
}

int spo2Index(int logicalIndex) {
  int start = (ppgWriteIdx - ppgCount + SPO2_WINDOW_SIZE) % SPO2_WINDOW_SIZE;
  return (start + logicalIndex) % SPO2_WINDOW_SIZE;
}

void updatePpgFs(unsigned long nowTs) {
  if (ppgLastTs == 0) {
    ppgLastTs = nowTs;
    ppgSampleCount = 0;
    return;
  }
  ppgSampleCount++;
  if (ppgSampleCount >= PPG_FS_MEASURE_WINDOW) {
    unsigned long elapsed = nowTs - ppgLastTs;
    if (elapsed > 0) {
      float measured = (float)ppgSampleCount * 1000.0f / (float)elapsed;
      if (measured > 30.0f && measured < 80.0f)
        ppgMeasuredFs = 0.8f * ppgMeasuredFs + 0.2f * measured;
    }
    ppgLastTs = nowTs;
    ppgSampleCount = 0;
  }
}

void readPpgStream() {
  max30102.check();
  while (max30102.available()) {
    uint32_t ir  = max30102.getFIFOIR();
    uint32_t red = max30102.getFIFORed();
    max30102.nextSample();

    unsigned long ts = millis();
    bool fingerDetected = ir > SPO2_MIN_IR && red > SPO2_MIN_RED;

    if (fingerDetected) {
      updatePpgFs(ts);
      spo2BadCount = 0;

      PpgSample p;
      p.ir    = ir;
      p.red   = red;
      p.ts    = ts;
      p.valid = !ppgGapFlag;
      ppgGapFlag = false;

      pushPpgSample(p);
    } else {
      spo2BadCount++;
      if (spo2BadCount >= SPO2_BAD_LIMIT) {
        resetPpgWindow();
        spo2BadCount = SPO2_BAD_LIMIT;
      }
    }
  }
}

bool checkPPGQuality(PPGQuality& q) {
  memset(&q, 0, sizeof(PPGQuality));

  if (ppgCount < SPO2_WINDOW_SIZE) {
    strcpy(q.reason, "NOT_ENOUGH_WINDOW");
    return false;
  }

  for (int i = 0; i < SPO2_WINDOW_SIZE; i++) {
    if (!ppgBuf[spo2Index(i)].valid) {
      strcpy(q.reason, "DISCONTINUOUS_SIGNAL");
      return false;
    }
  }

  unsigned long firstTs = ppgBuf[spo2Index(0)].ts;
  unsigned long lastTs  = ppgBuf[spo2Index(SPO2_WINDOW_SIZE - 1)].ts;
  unsigned long expectedDuration = (unsigned long)((SPO2_WINDOW_SIZE - 1) * 1000.0f / ppgMeasuredFs);
  unsigned long actualDuration   = lastTs - firstTs;

  if (actualDuration > expectedDuration * 1.25f || actualDuration < expectedDuration * 0.75f) {
    strcpy(q.reason, "IRREGULAR_TIMING");
    return false;
  }

  uint32_t irMin = 0xFFFFFFFF, irMax = 0;
  uint32_t redMin = 0xFFFFFFFF, redMax = 0;
  double irSum = 0, redSum = 0;
  float maxJerk = 0.0f;
  int badMotionCount = 0;

  for (int i = 0; i < SPO2_WINDOW_SIZE; i++) {
    const PpgSample& s = ppgBuf[spo2Index(i)];
    irSum  += s.ir;
    redSum += s.red;
    if (s.ir  < irMin)  irMin  = s.ir;
    if (s.ir  > irMax)  irMax  = s.ir;
    if (s.red < redMin) redMin = s.red;
    if (s.red > redMax) redMax = s.red;
  }

  int fallCount = cbCount();
  for (int i = 0; i < fallCount; i++) {
    const FallSample& f = fallBuf[(fallReadIdx + i) % CB_SIZE];
    if (f.ts < firstTs || f.ts > lastTs) continue;
    if (f.jerk > maxJerk) maxJerk = f.jerk;
    if (f.jerk > SPO2_MAX_JERK_SAMPLE) badMotionCount++;
  }

  q.irMean       = irSum / SPO2_WINDOW_SIZE;
  q.redMean      = redSum / SPO2_WINDOW_SIZE;
  q.irP2P        = irMax - irMin;
  q.redP2P       = redMax - redMin;
  q.irACDC       = q.irP2P  / q.irMean;
  q.redACDC      = q.redP2P / q.redMean;
  q.maxJerk      = maxJerk;
  q.badMotionRate = (float)badMotionCount / SPO2_WINDOW_SIZE;
  q.measuredFs   = ppgMeasuredFs;

  if (q.irMean < SPO2_MIN_IR)           { strcpy(q.reason, "LOW_IR_NO_FINGER");       return false; }
  if (q.redMean < SPO2_MIN_RED)         { strcpy(q.reason, "LOW_RED_NO_FINGER");      return false; }
  if (q.irP2P < SPO2_MIN_IR_P2P)        { strcpy(q.reason, "LOW_IR_AC_SIGNAL");       return false; }
  if (q.redP2P < SPO2_MIN_RED_P2P)      { strcpy(q.reason, "LOW_RED_AC_SIGNAL");      return false; }
  if (q.irACDC  < SPO2_MIN_ACDC || q.redACDC  < SPO2_MIN_ACDC) { strcpy(q.reason, "ACDC_TOO_LOW");         return false; }
  if (q.irACDC  > SPO2_MAX_ACDC || q.redACDC  > SPO2_MAX_ACDC) { strcpy(q.reason, "ACDC_TOO_HIGH_NOISY"); return false; }
  if (q.badMotionRate > SPO2_MAX_BAD_MOTION_RATE)               { strcpy(q.reason, "TOO_MUCH_MOTION");      return false; }

  strcpy(q.reason, "GOOD");
  q.ok = true;
  return true;
}

void updateBaseline(float totalG) {
  if (currentState != STATE_STABLE) return;
  baselineSumG += totalG;
  baselineCount++;
  if (baselineCount >= BASELINE_WINDOW) {
    float avg = baselineSumG / baselineCount;
    baselineG     = constrain(avg, 0.9f, 1.1f);
    baselineSumG  = 0.0f;
    baselineCount = 0;
  }
}

bool checkDelayedStillness(int peakIdx, float& stillRatioOut, float& meanGyroOut, float& meanJerkOut) {
  const int START_OFFSET = 30;
  const int CHECK_WINDOW = 50;
  int stillCount = 0;
  float sumGyro = 0.0f, sumJerk = 0.0f;
  for (int i = 0; i < CHECK_WINDOW; i++) {
    const FallSample& s = fallBuf[(peakIdx + START_OFFSET + i) % CB_SIZE];
    float gyroMag = sqrtf(s.gx*s.gx + s.gy*s.gy + s.gz*s.gz);
    sumGyro += gyroMag;
    sumJerk += s.jerk;
    if (s.jerk < 1.5f && gyroMag < 25.0f) stillCount++;
  }
  stillRatioOut = (float)stillCount / CHECK_WINDOW;
  meanGyroOut   = sumGyro / CHECK_WINDOW;
  meanJerkOut   = sumJerk / CHECK_WINDOW;
  return stillRatioOut >= 0.70f && meanGyroOut < 30.0f && meanJerkOut < 2.5f;
}

bool checkSlowFallPattern(int peakIdx, float& thetaRangeOut, float& avgGyroOut, float& finalThetaOut, float& finalAzOut) {
  const int BACK    = 80;
  const int FORWARD = 100;
  float minTheta = 999.0f, maxTheta = 0.0f, sumGyro = 0.0f;
  int count = 0;
  for (int i = -BACK; i < FORWARD; i++) {
    int idx = (peakIdx + i + CB_SIZE) % CB_SIZE;
    const FallSample& s = fallBuf[idx];
    if (s.theta < minTheta) minTheta = s.theta;
    if (s.theta > maxTheta) maxTheta = s.theta;
    float gyroMag = sqrtf(s.gx*s.gx + s.gy*s.gy + s.gz*s.gz);
    sumGyro += gyroMag;
    count++;
  }
  thetaRangeOut = maxTheta - minTheta;
  avgGyroOut    = sumGyro / count;
  const int FINAL_WINDOW = 30;
  float sumThetaFinal = 0.0f, sumAzFinal = 0.0f;
  for (int i = 0; i < FINAL_WINDOW; i++) {
    int idx = (peakIdx + FORWARD - FINAL_WINDOW + i + CB_SIZE) % CB_SIZE;
    const FallSample& s = fallBuf[idx];
    sumThetaFinal += s.theta;
    sumAzFinal    += fabsf(s.az);
  }
  finalThetaOut = sumThetaFinal / FINAL_WINDOW;
  finalAzOut    = sumAzFinal    / FINAL_WINDOW;
  return (thetaRangeOut >= 30.0f) && (avgGyroOut >= 20.0f && avgGyroOut <= 160.0f) &&
         (finalThetaOut >= 35.0f  || finalAzOut <= 0.70f);
}

bool preAIFilter(int peakIdx, float peakG, char* rejectReason, size_t rejectReasonLen) {
  if (cbCount() < THETA_PRE_WINDOW + THETA_POST_WINDOW) {
    strcpy(rejectReason, "REJECT_NOT_ENOUGH_DATA");
    return false;
  }

  float thetaBeforeArr[THETA_PRE_WINDOW];
  float thetaAfterArr[THETA_POST_WINDOW];
  for (int i = 0; i < THETA_PRE_WINDOW; i++)
    thetaBeforeArr[i] = fallBuf[(peakIdx - THETA_PRE_WINDOW + i + CB_SIZE) % CB_SIZE].theta;
  for (int i = 0; i < THETA_POST_WINDOW; i++)
    thetaAfterArr[i]  = fallBuf[(peakIdx + i) % CB_SIZE].theta;

  float thetaBefore = median(thetaBeforeArr, THETA_PRE_WINDOW);
  float thetaAfter  = median(thetaAfterArr,  THETA_POST_WINDOW);
  float deltaTheta  = fabsf(thetaAfter - thetaBefore);

  float sumG = 0.0f, sumG2 = 0.0f, sumGyro = 0.0f, maxGyro = 0.0f;
  int stillCount = 0;
  for (int i = 0; i < VAR_G_POST_WINDOW; i++) {
    const FallSample& s = fallBuf[(peakIdx + i) % CB_SIZE];
    float g       = s.mag;
    float gyroMag = sqrtf(s.gx*s.gx + s.gy*s.gy + s.gz*s.gz);
    sumG   += g;
    sumG2  += g * g;
    sumGyro += gyroMag;
    if (gyroMag > maxGyro) maxGyro = gyroMag;
    if (s.jerk < FALL_JERK_STILL && gyroMag < FALL_GYRO_STILL) stillCount++;
  }

  float meanG   = sumG  / VAR_G_POST_WINDOW;
  float varG    = (sumG2 / VAR_G_POST_WINDOW) - (meanG * meanG);
  if (varG < 0.0f) varG = 0.0f;
  float meanGyro    = sumGyro / VAR_G_POST_WINDOW;
  float stillRatio  = (float)stillCount / VAR_G_POST_WINDOW;

  float delayedStillRatio = 0.0f, delayedMeanGyro = 0.0f, delayedMeanJerk = 0.0f;
  bool delayedStill = checkDelayedStillness(peakIdx, delayedStillRatio, delayedMeanGyro, delayedMeanJerk);

  float slowThetaRange = 0.0f, slowAvgGyro = 0.0f, slowFinalTheta = 0.0f, slowFinalAz = 0.0f;
  bool slowFall = checkSlowFallPattern(peakIdx, slowThetaRange, slowAvgGyro, slowFinalTheta, slowFinalAz);

  bool violentFallLike        = peakG >= 2.7f && maxGyro >= 350.0f;
  bool hardRotatingFallLike   = peakG >= 2.5f && maxGyro >= 300.0f && delayedMeanGyro >= 60.0f && delayedMeanJerk >= 4.0f;

  bool activeADL = peakG < 2.4f && maxGyro < 280.0f && meanGyro >= 30.0f &&
                   delayedStillRatio < 0.35f && stillRatio < 0.45f &&
                   delayedMeanGyro >= 25.0f && delayedMeanJerk >= 2.0f;
  bool runningLike = peakG >= 1.4f && peakG < 2.4f && maxGyro >= 70.0f && maxGyro <= 260.0f &&
                     meanGyro >= 30.0f && delayedStillRatio < 0.35f &&
                     delayedMeanGyro >= 25.0f && delayedMeanJerk >= 2.0f;
  bool stairLikePost = peakG >= 1.2f && peakG < 2.4f && maxGyro < 260.0f &&
                       meanGyro >= 20.0f && delayedStillRatio < 0.40f &&
                       stillRatio < 0.50f && delayedMeanJerk >= 1.8f;
  bool jumpLandingLikePost = peakG >= 1.8f && peakG < 2.6f && maxGyro < 300.0f &&
                             delayedStillRatio < 0.45f && stillRatio < 0.50f &&
                             delayedMeanGyro >= 20.0f && delayedMeanJerk >= 1.8f;

  if (!violentFallLike && !hardRotatingFallLike &&
      (activeADL || runningLike || stairLikePost || jumpLandingLikePost)) {
    snprintf(rejectReason, rejectReasonLen,
             "REJECT_ADL_ACTIVE: peak=%.2fg dTheta=%.1f meanGyro=%.1f maxGyro=%.1f still=%.2f delayedStill=%.2f delayedGyro=%.1f delayedJerk=%.2f",
             peakG, deltaTheta, meanGyro, maxGyro, stillRatio,
             delayedStillRatio, delayedMeanGyro, delayedMeanJerk);
    return false;
  }

  bool strongImpact  = peakG >= FALL_STRONG_PEAK_G;
  bool mediumImpact  = peakG >= FALL_MEDIUM_PEAK_G;
  bool weakImpact    = peakG >= FALL_MIN_PEAK_G;
  bool thetaStrong   = deltaTheta >= FALL_THETA_STRONG;
  bool thetaSoft     = deltaTheta >= FALL_THETA_SOFT;
  bool postStill     = stillRatio >= FALL_STILL_RATIO  && meanGyro < FALL_GYRO_STILL;
  bool veryStill     = stillRatio >= FALL_VERY_STILL_RATIO && meanGyro < 18.0f;
  bool rotationStrong = maxGyro >= FALL_GYRO_ROTATE;
  bool chairLike     = peakG >= 1.8f && maxGyro >= FALL_GYRO_CHAIR &&
                       deltaTheta >= 25.0f && stillRatio >= 0.80f;

  int score = 0;
  if (strongImpact) score += 3; else if (mediumImpact) score += 2; else if (weakImpact) score += 1;
  if (thetaStrong)  score += 2; else if (thetaSoft)   score += 1;
  if (slowFall)     score += 3;
  if (postStill)    score += 2;
  if (veryStill)    score += 1;
  if (delayedStill) score += 3;
  if (rotationStrong) score += 1;
  if (chairLike)    score += 2;
  if (freeFallDetected) score += 2;

  bool fallWithStillness        = strongImpact && deltaTheta >= 20.0f && delayedStill;
  bool fallWithMovement         = strongImpact && peakG >= 2.5f && rotationStrong && deltaTheta >= 25.0f;
  bool mediumFallWithMovement   = mediumImpact && peakG >= 1.8f && rotationStrong && deltaTheta >= 35.0f && score >= FALL_SCORE_PASS;
  bool chairOrSideFall          = peakG >= 1.6f && maxGyro >= 140.0f && deltaTheta >= 15.0f &&
                                  (delayedStillRatio >= 0.20f || meanGyro >= 30.0f);
  bool chairOrSideFallMoving    = peakG >= 1.9f && maxGyro >= 200.0f && deltaTheta >= 20.0f &&
                                  delayedStillRatio >= 0.25f && delayedMeanGyro < 45.0f;
  bool slowSupportedFallMoving  = peakG >= 1.25f && peakG < 2.4f && slowFall &&
                                  slowThetaRange >= 40.0f && slowAvgGyro >= 25.0f && slowAvgGyro <= 180.0f;

  bool pass = fallWithStillness || fallWithMovement || mediumFallWithMovement ||
              chairOrSideFall || chairOrSideFallMoving || slowSupportedFallMoving ||
              violentFallLike || hardRotatingFallLike;

  if (!pass) {
    snprintf(rejectReason, rejectReasonLen,
             "REJECT_SCORE: score=%d peak=%.2fg dTheta=%.1f varG=%.3f meanGyro=%.1f maxGyro=%.1f still=%.2f delayedStill=%.2f delayedGyro=%.1f delayedJerk=%.2f",
             score, peakG, deltaTheta, varG, meanGyro, maxGyro, stillRatio,
             delayedStillRatio, delayedMeanGyro, delayedMeanJerk);
    return false;
  }

  snprintf(rejectReason, rejectReasonLen,
           "PASS_SCORE: score=%d peak=%.2fg dTheta=%.1f varG=%.3f meanGyro=%.1f maxGyro=%.1f still=%.2f delayedStill=%.2f delayedGyro=%.1f delayedJerk=%.2f",
           score, peakG, deltaTheta, varG, meanGyro, maxGyro, stillRatio,
           delayedStillRatio, delayedMeanGyro, delayedMeanJerk);
  return true;
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, payload, length)) return;
  const char* cmd = doc["cmd"] | "";
  if      (strcmp(cmd, "buzzer_on")  == 0) buzzerBip(doc["duration_ms"] | BUZZER_CMD_MS);
  else if (strcmp(cmd, "buzzer_off") == 0) { digitalWrite(BUZZER_PIN, LOW); buzzerOn = false; }
  else if (strcmp(cmd, "reboot")     == 0) { delay(500); ESP.restart(); }
}

void setupWiFi() {
  WiFi.mode(WIFI_STA);
  for (auto const& wifi : WIFI_NETWORKS) {
    const char* ssid = wifi.first.c_str();
    const char* pass = wifi.second.c_str();
    Serial.printf("\n[WIFI] Ket noi: %s\n", ssid);
    WiFi.begin(ssid, pass);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts <= 20) {
      delay(500);
      Serial.print(".");
      attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("\n[WIFI] OK! IP: %s\n", WiFi.localIP().toString().c_str());
      return;
    }
    Serial.printf("\n[WIFI] That bai: %s\n", ssid);
    WiFi.disconnect(true);
    delay(500);
  }
  Serial.println("\n[WIFI] Khong ket noi duoc WiFi nao!");
}

void reconnectMQTT() {
  if (mqttClient.connected()) return;
  Serial.print("[MQTT] Ket noi lai...");
  String cid = "ESP32_HM_" + String(random(99999));
  if (mqttClient.connect(cid.c_str(), MQTT_USER, MQTT_PASS)) {
    Serial.println(" OK!");
    mqttClient.subscribe(TOPIC_COMMAND);
  } else {
    Serial.printf(" FAIL (%d). Thu lai sau 3s.\n", mqttClient.state());
    delay(3000);
  }
}

void sendFallAlert(const char* reason, float peakG, float deltaG) {
  char buf[192];
  snprintf(buf, sizeof(buf),
    "{\"status\":\"checking\",\"trigger\":\"%s\",\"peak_g\":%.2f,\"delta_g\":%.2f}",
    reason, peakG, deltaG);
  mqttClient.publish(TOPIC_ALERT, buf, false);
  Serial.printf("[ALERT] %s | peak=%.2fg delta=%.2fg\n", reason, peakG, deltaG);
}

void sendRejectAlert(const char* rejectReason, float peakG) {
  char buf[512];
  snprintf(buf, sizeof(buf),
    "{\"status\":\"rejected\",\"reason\":\"%s\",\"peak_g\":%.2f,\"total_rejected\":%lu}",
    rejectReason, peakG, (unsigned long)filterRejectCount);
  mqttClient.publish(TOPIC_ALERT, buf, false);
  Serial.printf("[FILTER] BI LOAI: %s\n", rejectReason);
}

void sendFallBinary() {
  char filterLog[384] = "";
  bool passed = preAIFilter(peakWriteIdx, transitionPeakG, filterLog, sizeof(filterLog));
  if (!passed) {
    filterRejectCount++;
    Serial.printf("[FILTER] %s | Tong reject: %lu\n", filterLog, (unsigned long)filterRejectCount);
    if (mqttClient.connected()) sendRejectAlert(filterLog, transitionPeakG);
    return;
  }

  filterPassCount++;
  lastFallAlert = millis();
  sendFallAlert(triggerReason, transitionPeakG, transitionDeltaG);
  Serial.printf("[FILTER] %s | Tong pass: %lu\n", filterLog, (unsigned long)filterPassCount);

  peakWriteIdx  = peakWriteIdx % CB_SIZE;
  int preAvail  = (peakWriteIdx - fallReadIdx + CB_SIZE) % CB_SIZE;
  int actualPre = min(preAvail, PRE_SAMPLES);
  int actualPost = min(POST_SAMPLES, cbCount() - actualPre);
  if (actualPost < 0) actualPost = 0;
  int totalSend = actualPre + actualPost;
  if (totalSend == 0) { Serial.println("[AI] Khong du du lieu!"); return; }

  uint8_t* ptr = binaryBuf;
  ptr = writeU32BE(ptr, (uint32_t)triggerTime);
  ptr = writeU16BE(ptr, (uint16_t)totalSend);
  ptr = writeU8(ptr,   (uint8_t)actualPre);
  uint8_t rlen = (uint8_t)strnlen(triggerReason, REASON_MAX_LEN - 1);
  ptr = writeU8(ptr, rlen);
  memcpy(ptr, triggerReason, rlen); ptr += rlen;

  int dataStart = (peakWriteIdx - actualPre + CB_SIZE) % CB_SIZE;
  for (int i = 0; i < totalSend; i++) {
    const FallSample& s = fallBuf[(dataStart + i) % CB_SIZE];
    ptr = writeI16BE(ptr, floatToI16(s.ax,   SCALE_ACC));
    ptr = writeI16BE(ptr, floatToI16(s.ay,   SCALE_ACC));
    ptr = writeI16BE(ptr, floatToI16(s.az,   SCALE_ACC));
    ptr = writeI16BE(ptr, floatToI16(s.gx,   SCALE_GYRO));
    ptr = writeI16BE(ptr, floatToI16(s.gy,   SCALE_GYRO));
    ptr = writeI16BE(ptr, floatToI16(s.gz,   SCALE_GYRO));
    ptr = writeI16BE(ptr, floatToI16(s.mag,  SCALE_MAG));
    ptr = writeI16BE(ptr, floatToI16(s.jerk, SCALE_JERK));
  }

  size_t sz = ptr - binaryBuf;
  if (sz > BINARY_BUF_SIZE) { Serial.printf("[AI] BUFFER OVERFLOW! sz=%d\n", sz); return; }
  Serial.printf("[AI] Binary: %d bytes | %d pre + %d post\n", sz, actualPre, actualPost);
  if (mqttClient.connected()) {
    bool ok = mqttClient.publish(TOPIC_FALL_RAW, binaryBuf, sz, false);
    Serial.printf("[AI] Publish %s!\n", ok ? "OK" : "FAIL");
  } else {
    Serial.println("[AI] MQTT ngat ket noi!");
  }
}

void sendSensorWindow() {
  static unsigned long lastWaitLog = 0;

  if (ppgCount < SPO2_WINDOW_SIZE) {
    if (millis() - lastWaitLog >= 1000) {
      lastWaitLog = millis();
      Serial.printf("[SPO2] Chua du mau: count=%d/%d new=%d/%d MQTT=%s\n",
                    ppgCount, SPO2_WINDOW_SIZE, ppgNewSamples, SPO2_STEP_SIZE,
                    mqttClient.connected() ? "OK" : "NO");
    }
    return;
  }

  if (ppgNewSamples < SPO2_STEP_SIZE) return;

  PPGQuality q;
  bool goodSignal = checkPPGQuality(q);
  if (!goodSignal) {
    Serial.printf("[SPO2] Quality xau, khong gui: %s\n", q.reason);
    ppgNewSamples = 0;
    return;
  }

  const char* postureStr =
    currentState == STATE_POSTCAPTURE ? "post_fall" :
    currentState == STATE_CONFIRMING  ? "confirming" :
    currentState == STATE_TRANSITION  ? "transition" : "stable";

  float currentAz = 0.0f;
  float currentTheta = 0.0f;
  if (cbCount() > 0) {
    const FallSample& last = fallBuf[(fallWriteIdx - 1 + CB_SIZE) % CB_SIZE];
    currentAz    = fabsf(last.az);
    currentTheta = last.theta;
  }

  char* ptr    = jsonBuf;
  char* endBuf = jsonBuf + JSON_BUF_SIZE - 512;

  ptr += sprintf(ptr,
    "{\"type\":\"ppg_window\",\"fs\":%.2f,\"fs_nominal\":%d,"
    "\"window_size\":%d,\"step_size\":%d,\"seq\":%lu,"
    "\"temp\":%.2f,\"posture\":\"%s\",\"az\":%.3f,\"theta\":%.1f,"
    "\"quality\":{\"status\":\"%s\",\"valid\":%s,"
    "\"ir_mean\":%.0f,\"red_mean\":%.0f,"
    "\"ir_p2p\":%.0f,\"red_p2p\":%.0f,"
    "\"ir_acdc\":%.5f,\"red_acdc\":%.5f,"
    "\"max_jerk\":%.2f,\"bad_motion_rate\":%.3f},"
    "\"data\":[",
    q.measuredFs,
    PPG_SENSOR_SAMPLE_RATE_HZ,
    SPO2_WINDOW_SIZE,
    SPO2_STEP_SIZE,
    (unsigned long)ppgSeqOut++,
    temperature,
    postureStr,
    currentAz,
    currentTheta,
    q.reason,
    goodSignal ? "true" : "false",
    q.irMean, q.redMean,
    q.irP2P,  q.redP2P,
    q.irACDC, q.redACDC,
    q.maxJerk, q.badMotionRate
  );

  for (int i = 0; i < SPO2_WINDOW_SIZE; i++) {
    if (ptr >= endBuf) { Serial.println("[SPO2] JSON buffer gan day, dung ghi mau"); break; }
    const PpgSample& s = ppgBuf[spo2Index(i)];
    ptr += sprintf(ptr,
      (i < SPO2_WINDOW_SIZE - 1)
        ? "{\"t\":%lu,\"ir\":%lu,\"red\":%lu,\"v\":%d},"
        : "{\"t\":%lu,\"ir\":%lu,\"red\":%lu,\"v\":%d}",
      s.ts, (unsigned long)s.ir, (unsigned long)s.red, (int)s.valid
    );
  }

  ptr += sprintf(ptr, "]}");
  int payloadSize = ptr - jsonBuf;

  if (!mqttClient.connected()) {
    Serial.println("[SPO2] MQTT chua ket noi, khong gui duoc");
    return;
  }

  Serial.printf("[SPO2] Gui MQTT size=%d fs=%.1fHz posture=%s\n", payloadSize, q.measuredFs, postureStr);
  bool ok = mqttClient.publish(TOPIC_SENSOR, (uint8_t*)jsonBuf, payloadSize, false);
  Serial.printf("[SPO2] MQTT publish: %s\n", ok ? "OK" : "FAIL");

  if (ok) ppgNewSamples = 0;
}

void updateStateMachine(float totalG, float az, unsigned long now) {
  float deltaG = fabsf(totalG - baselineG);

  switch (currentState) {

    case STATE_STABLE: {
      if (totalG < 0.5f) { freeFallDetected = true; freeFallTime = now; }

      if (now - windowStart > DELTA_G_WINDOW_MS) {
        peakGInWindow = totalG; maxDeltaG = deltaG; windowStart = now;
      } else {
        if (totalG > peakGInWindow) { peakGInWindow = totalG; peakWriteIdx = (fallWriteIdx - 1 + CB_SIZE) % CB_SIZE; }
        if (deltaG > maxDeltaG) maxDeltaG = deltaG;
      }

      if (peakGInWindow > PEAK_G_INSTANT) {
        transitionPeakG  = peakGInWindow;
        transitionDeltaG = maxDeltaG;
        snprintf(triggerReason, sizeof(triggerReason), "hard_fall_peak_%.1fg", peakGInWindow);
        Serial.printf("[FSM] Te cung! Peak=%.2fg -> CONFIRMING.\n", peakGInWindow);
        currentState = STATE_CONFIRMING; stateEnteredMs = now;
        peakGInWindow = 0; maxDeltaG = 0; windowStart = now;
        break;
      }

      if (maxDeltaG > DELTA_G_THRESHOLD || (freeFallDetected && peakGInWindow > 1.5f)) {
        transitionPeakG  = peakGInWindow;
        transitionDeltaG = maxDeltaG;
        snprintf(triggerReason, sizeof(triggerReason),
                 freeFallDetected ? "freefall_impact" : "transition_delta_%.1fg", maxDeltaG);
        currentState = STATE_TRANSITION; stateEnteredMs = now;
        peakGInWindow = 0; maxDeltaG = 0; windowStart = now;
      }

      if (freeFallDetected && (now - freeFallTime > 1000)) freeFallDetected = false;
      break;
    }

    case STATE_TRANSITION: {
      if (totalG > transitionPeakG) {
        transitionPeakG = totalG;
        peakWriteIdx    = (fallWriteIdx - 1 + CB_SIZE) % CB_SIZE;
      }
      if (now - stateEnteredMs < CONFIRM_WAIT_MS) break;

      float azAbs = fabsf(az);
      float maxGyroRecent = 0.0f, minAzRecent = 10.0f, maxThetaRecent = 0.0f, minThetaRecent = 999.0f;
      const int CHECK_BACK = 35;
      for (int i = 0; i < CHECK_BACK; i++) {
        int idx = (fallWriteIdx - 1 - i + CB_SIZE) % CB_SIZE;
        const FallSample& s = fallBuf[idx];
        float gmag = sqrtf(s.gx*s.gx + s.gy*s.gy + s.gz*s.gz);
        if (gmag       > maxGyroRecent)  maxGyroRecent  = gmag;
        if (fabsf(s.az)< minAzRecent)    minAzRecent    = fabsf(s.az);
        if (s.theta    > maxThetaRecent) maxThetaRecent = s.theta;
        if (s.theta    < minThetaRecent) minThetaRecent = s.theta;
      }
      float thetaSwingRecent = maxThetaRecent - minThetaRecent;
      bool uprightAfter = azAbs > 0.70f && minAzRecent > 0.60f;

      bool stairLike = transitionPeakG >= 1.15f && transitionPeakG <= 2.4f &&
                       maxGyroRecent < 220.0f && thetaSwingRecent < 30.0f && uprightAfter && !freeFallDetected;
      bool stepUpDownLike = transitionPeakG >= 1.25f && transitionPeakG <= 2.6f &&
                            maxGyroRecent < 260.0f && thetaSwingRecent < 35.0f && uprightAfter && !freeFallDetected;
      bool jumpLandingLike = transitionPeakG >= 1.8f && transitionPeakG <= 3.4f &&
                             maxGyroRecent < 300.0f && thetaSwingRecent < 30.0f && uprightAfter && !freeFallDetected;

      if (stairLike || stepUpDownLike || jumpLandingLike) {
        Serial.printf("[FSM] ADL reject: peak=%.2f |az|=%.2f -> STABLE.\n", transitionPeakG, azAbs);
        currentState = STATE_STABLE; memset(triggerReason, 0, sizeof(triggerReason));
        break;
      }

      bool postureMaybeFall = azAbs < 0.60f || minAzRecent < 0.55f || thetaSwingRecent >= 40.0f;
      bool bodyGNormal      = totalG < 1.35f;
      bool strongImpact     = transitionPeakG >= 2.4f;
      bool mediumImpact     = transitionPeakG >= 1.75f;
      bool rotateStrong     = maxGyroRecent >= 130.0f;

      bool strongImpactFall   = strongImpact && (azAbs < 0.60f || minAzRecent < 0.55f || thetaSwingRecent >= 40.0f || freeFallDetected);
      bool chairCandidate     = transitionPeakG >= 1.8f && maxGyroRecent >= 180.0f && thetaSwingRecent >= 35.0f && minAzRecent < 0.60f && azAbs < 0.65f;
      bool slowFallCandidate  = transitionPeakG >= 1.35f && transitionPeakG < 2.2f && thetaSwingRecent >= 60.0f && maxGyroRecent >= 50.0f && maxGyroRecent <= 200.0f && minAzRecent < 0.55f && azAbs < 0.55f;

      if (strongImpactFall || (mediumImpact && postureMaybeFall && bodyGNormal && rotateStrong) || chairCandidate || slowFallCandidate) {
        snprintf(triggerReason, sizeof(triggerReason),
                 "suspected_fall_peak%.2f_az%.2f_maxGyro%.1f_thetaSwing%.1f",
                 transitionPeakG, azAbs, maxGyroRecent, thetaSwingRecent);
        Serial.printf("[FSM] Nghi te! peak=%.2f -> CONFIRMING.\n", transitionPeakG);
        currentState = STATE_CONFIRMING; stateEnteredMs = now;
      } else {
        Serial.printf("[FSM] ADL binh thuong -> STABLE.\n");
        currentState = STATE_STABLE; memset(triggerReason, 0, sizeof(triggerReason));
      }
      break;
    }

    case STATE_CONFIRMING: {
      if (now - lastFallAlert < FALL_COOLDOWN_MS) {
        Serial.println("[FSM] Cooldown chua het -> bo qua.");
        currentState = STATE_STABLE; break;
      }
      triggerTime     = now;
      postSampleCount = 0;
      currentState    = STATE_POSTCAPTURE;
      stateEnteredMs  = now;
      Serial.printf("[FSM] Thu them %d mau post-peak...\n", POST_SAMPLES);
      break;
    }

    case STATE_POSTCAPTURE:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================");
  Serial.println("  FALL DETECTION v10");
  Serial.printf("  Window: %d pre + %d post = %d mau\n", PRE_SAMPLES, POST_SAMPLES, LSTM_WINDOW_SIZE);
  Serial.printf("  Channels: %d | JSON_BUF: %d bytes\n", NUM_CHANNELS, JSON_BUF_SIZE);
  Serial.println("  Baseline window: 150 mau (1.5s)");
  Serial.println("  PPG: do fs thuc te, kiem tra lien tuc, nhan posture");
  Serial.println("  Nhiet do: EMA alpha=0.07 + offset=1.5C");
  Serial.println("========================================");

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Wire.begin(21, 22);    Wire.setClock(400000);
  I2C_MAX.begin(32, 33); I2C_MAX.setClock(400000);

  if (!max30102.begin(I2C_MAX, I2C_SPEED_FAST)) {
    Serial.println("[SETUP] MAX30102 LOI!");
  } else {
    max30102.setup(0x40, PPG_FIFO_SAMPLE_AVERAGE, 2, PPG_SENSOR_SAMPLE_RATE_HZ, 411, 8192);
    max30102.setPulseAmplitudeRed(0x40);
    max30102.setPulseAmplitudeIR(0x40);
    Serial.println("[SETUP] MAX30102 OK");
  }

  max30205.begin(0x48);
  Serial.println("[SETUP] MAX30205 OK");

  mpu6050.begin();
  mpu6050.setGyroOffsets(0.00, 0.00, 0.00);
  Serial.println("[SETUP] MPU6050 OK");

  setupWiFi();
  espClient.setInsecure();
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setBufferSize(20000);
  mqttClient.setKeepAlive(30);
  mqttClient.setSocketTimeout(10);

  jsonBuf = (char*)malloc(JSON_BUF_SIZE);
  if (!jsonBuf) { Serial.println("[SETUP] Khong du RAM cho jsonBuf!"); while (1); }

  windowStart  = millis();
  lastSample   = millis();
  prevMag      = 1.0f;
  ppgGapFlag   = true;

  float initTemp = max30205.readTemperature();
  if (!isnan(initTemp) && initTemp > 20.0f && initTemp < 45.0f) {
    temperature     = initTemp + TEMP_SKIN_OFFSET;
    tempInitialized = true;
  }

  Serial.printf("[MEM] Free heap: %d bytes\n", ESP.getFreeHeap());

  digitalWrite(BUZZER_PIN, HIGH); delay(100); digitalWrite(BUZZER_PIN, LOW); delay(100);
  digitalWrite(BUZZER_PIN, HIGH); delay(100); digitalWrite(BUZZER_PIN, LOW);

  Serial.println("[SYSTEM] KHOI TAO THANH CONG!\n");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WIFI] Mat ket noi! Ket noi lai...");
    setupWiFi();
  }
  if (!mqttClient.connected()) reconnectMQTT();
  mqttClient.loop();
  buzzerUpdate();
  readPpgStream();

  if (filterPassCount   > 1000000) filterPassCount   = 0;
  if (filterRejectCount > 1000000) filterRejectCount = 0;

  if (millis() - lastTempRead >= 1000) {
    float t = max30205.readTemperature();
    if (!isnan(t) && t > 20.0f && t < 45.0f) {
      float corrected = t + TEMP_SKIN_OFFSET;
      if (!tempInitialized) {
        temperature     = corrected;
        tempInitialized = true;
      } else {
        temperature = (1.0f - TEMP_EMA_ALPHA) * temperature + TEMP_EMA_ALPHA * corrected;
      }
    }
    lastTempRead = millis();
  }

  if (millis() - lastSample >= SAMPLE_INTERVAL_MS) {
    unsigned long now = lastSample;
    lastSample = millis();
    mpu6050.update();

    FallSample s;
    s.ts    = now;
    s.ax    = mpu6050.getAccX();
    s.ay    = mpu6050.getAccY();
    s.az    = mpu6050.getAccZ();
    s.gx    = mpu6050.getGyroX();
    s.gy    = mpu6050.getGyroY();
    s.gz    = mpu6050.getGyroZ();
    s.mag   = sqrtf(s.ax*s.ax + s.ay*s.ay + s.az*s.az);
    s.jerk  = fabsf(s.mag - prevMag) / (SAMPLE_INTERVAL_MS / 1000.0f);
    s.theta = calcTheta(s.ax, s.ay, s.az);
    prevMag = s.mag;

    cbPush(s);
    updateBaseline(s.mag);

    if (now > 5000) updateStateMachine(s.mag, s.az, now);

    if (currentState == STATE_POSTCAPTURE) {
      postSampleCount++;
      if (postSampleCount >= POST_SAMPLES) {
        Serial.printf("[FSM] Du %d mau. Kiem tra bo loc...\n", POST_SAMPLES);
        sendFallBinary();
        currentState = STATE_STABLE;
        Serial.println("[FSM] Tro ve STABLE.");
      }
    }

    sendSensorWindow();
  }

  if (millis() - lastLog >= 2000) {
    lastLog = millis();
    mpu6050.update();
    float ax = mpu6050.getAccX(), ay = mpu6050.getAccY(), az = mpu6050.getAccZ();
    float mag   = sqrtf(ax*ax + ay*ay + az*az);
    float theta = calcTheta(ax, ay, az);

    const char* stateStr =
      currentState == STATE_STABLE     ? "STABLE"     :
      currentState == STATE_TRANSITION ? "TRANSITION" :
      currentState == STATE_CONFIRMING ? "CONFIRMING" : "POSTCAPTURE";

    Serial.println("----------------------------------------");
    Serial.printf("[STATUS] WiFi:%-4s MQTT:%-4s State:%s\n",
                  WiFi.status()==WL_CONNECTED ? "OK" : "NGAT",
                  mqttClient.connected()      ? "OK" : "NGAT", stateStr);
    Serial.printf("[SENSOR] G:%.3fg |az|:%.3fg th:%.1fdeg Base:%.3fg\n",
                  mag, fabsf(az), theta, baselineG);
    Serial.printf("[SENSOR] Temp:%.1fC | Post:%d/%d\n",
                  temperature,
                  currentState == STATE_POSTCAPTURE ? postSampleCount : 0,
                  POST_SAMPLES);
    Serial.printf("[PPG] fs_measured:%.1fHz gap:%d seq:%lu new:%d/%d\n",
                  ppgMeasuredFs, (int)ppgGapFlag, (unsigned long)ppgSeqOut,
                  ppgNewSamples, SPO2_STEP_SIZE);
    Serial.printf("[FILTER] Pass:%lu Reject:%lu\n",
                  (unsigned long)filterPassCount, (unsigned long)filterRejectCount);
    Serial.printf("[MEM] Free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.println("----------------------------------------");
  }
}
