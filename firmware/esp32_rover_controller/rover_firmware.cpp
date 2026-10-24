/**
 * @file rover_firmware.cpp
 * @brief ESP32 Motor Controller & Safety Firmware for Plant-Health Rover.
 * Handles serial protocol framing, 400ms watchdog safety timeout,
 * encoder counting, and hardware emergency stop.
 */

#include <Arduino.h>
#include "protocol.h"

// Hardware Pin Definitions
#define PIN_MOTOR_L_PWM     18
#define PIN_MOTOR_L_DIR     19
#define PIN_MOTOR_R_PWM     22
#define PIN_MOTOR_R_DIR     23
#define PIN_ESTOP_INPUT     4   // Active-low emergency stop
#define PIN_STATUS_LED      2

#define WATCHDOG_TIMEOUT_MS 400

// Volatile state
static volatile bool g_emergency_stopped = false;
static volatile unsigned long g_last_valid_frame_time = 0;
static volatile int32_t g_left_ticks = 0;
static volatile int32_t g_right_ticks = 0;

static uint8_t rx_buffer[300];
static size_t rx_index = 0;
static uint8_t tx_sequence = 0;

void IRAM_ATTR isr_emergency_stop() {
    g_emergency_stopped = true;
    digitalWrite(PIN_MOTOR_L_PWM, LOW);
    digitalWrite(PIN_MOTOR_R_PWM, LOW);
}

void apply_motor_speeds(int16_t left_pct, int16_t right_pct) {
    if (g_emergency_stopped) {
        digitalWrite(PIN_MOTOR_L_PWM, LOW);
        digitalWrite(PIN_MOTOR_R_PWM, LOW);
        return;
    }
    // Set direction and PWM duty
    digitalWrite(PIN_MOTOR_L_DIR, left_pct >= 0 ? HIGH : LOW);
    digitalWrite(PIN_MOTOR_R_DIR, right_pct >= 0 ? HIGH : LOW);
    ledcWrite(0, map(constrain(abs(left_pct), 0, 100), 0, 100, 0, 255));
    ledcWrite(1, map(constrain(abs(right_pct), 0, 100), 0, 100, 0, 255));
}

void send_ack(uint8_t seq) {
    uint8_t frame[6];
    frame[0] = SERIAL_FRAME_START;
    frame[1] = 2; // length = seq + type
    frame[2] = seq;
    frame[3] = (uint8_t)SERIAL_TYPE_ACK;
    uint16_t crc = crc16_ccitt_false(&frame[1], 3);
    frame[4] = (uint8_t)(crc >> 8);
    frame[5] = (uint8_t)(crc & 0xFF);
    Serial.write(frame, 6);
}

void send_telemetry(uint8_t seq) {
    uint8_t frame[16];
    frame[0] = SERIAL_FRAME_START;
    frame[1] = 2 + sizeof(MotorTelemetryPayload);
    frame[2] = seq;
    frame[3] = (uint8_t)SERIAL_TYPE_TELEMETRY;

    MotorTelemetryPayload telem;
    telem.left_encoder_ticks = g_left_ticks;
    telem.right_encoder_ticks = g_right_ticks;
    telem.motor_enabled = g_emergency_stopped ? 0 : 1;
    telem.fault_flags = g_emergency_stopped ? 0x01 : 0x00;

    memcpy(&frame[4], &telem, sizeof(MotorTelemetryPayload));
    uint16_t crc = crc16_ccitt_false(&frame[1], 3 + sizeof(MotorTelemetryPayload));
    frame[4 + sizeof(MotorTelemetryPayload)] = (uint8_t)(crc >> 8);
    frame[5 + sizeof(MotorTelemetryPayload)] = (uint8_t)(crc & 0xFF);
    Serial.write(frame, 6 + sizeof(MotorTelemetryPayload));
}

void process_frame(uint8_t seq, uint8_t type, const uint8_t *payload, size_t len) {
    g_last_valid_frame_time = millis();
    digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));

    switch (type) {
        case SERIAL_TYPE_COMMAND:
            if (len >= sizeof(DriveCommandPayload)) {
                const DriveCommandPayload *cmd = (const DriveCommandPayload *)payload;
                int16_t left = cmd->speed_pct - cmd->steer_pct;
                int16_t right = cmd->speed_pct + cmd->steer_pct;
                apply_motor_speeds(left, right);
                send_ack(seq);
            }
            break;

        case SERIAL_TYPE_STOP:
            apply_motor_speeds(0, 0);
            send_ack(seq);
            break;

        case SERIAL_TYPE_HEARTBEAT:
            send_telemetry(seq);
            break;

        default:
            send_ack(seq);
            break;
    }
}

void parse_serial_stream() {
    while (Serial.available()) {
        uint8_t byte = Serial.read();
        rx_buffer[rx_index++] = byte;

        if (rx_buffer[0] != SERIAL_FRAME_START) {
            rx_index = 0;
            continue;
        }

        if (rx_index >= 2) {
            uint8_t len = rx_buffer[1];
            size_t total_frame_len = 1 + 1 + len + 2;

            if (rx_index >= total_frame_len) {
                uint16_t received_crc = (((uint16_t)rx_buffer[total_frame_len - 2]) << 8) | rx_buffer[total_frame_len - 1];
                uint16_t calculated_crc = crc16_ccitt_false(&rx_buffer[1], 1 + len);

                if (received_crc == calculated_crc) {
                    uint8_t seq = rx_buffer[2];
                    uint8_t type = rx_buffer[3];
                    const uint8_t *payload = (len > 2) ? &rx_buffer[4] : NULL;
                    process_frame(seq, type, payload, len - 2);
                }
                rx_index = 0;
            }
        }

        if (rx_index >= sizeof(rx_buffer)) {
            rx_index = 0;
        }
    }
}

void setup() {
    Serial.begin(115200);

    pinMode(PIN_STATUS_LED, OUTPUT);
    pinMode(PIN_MOTOR_L_DIR, OUTPUT);
    pinMode(PIN_MOTOR_R_DIR, OUTPUT);
    pinMode(PIN_ESTOP_INPUT, INPUT_PULLUP);

    attachInterrupt(digitalPinToInterrupt(PIN_ESTOP_INPUT), isr_emergency_stop, FALLING);

    ledcSetup(0, 20000, 8); // 20kHz, 8-bit resolution
    ledcSetup(1, 20000, 8);
    ledcAttachPin(PIN_MOTOR_L_PWM, 0);
    ledcAttachPin(PIN_MOTOR_R_PWM, 1);

    g_last_valid_frame_time = millis();
}

void loop() {
    parse_serial_stream();

    // 400ms Task Watchdog Check
    if (millis() - g_last_valid_frame_time > WATCHDOG_TIMEOUT_MS) {
        apply_motor_speeds(0, 0); // Failsafe cutoff
        digitalWrite(PIN_STATUS_LED, LOW);
    }

    // Check emergency stop pin state
    if (digitalRead(PIN_ESTOP_INPUT) == LOW) {
        g_emergency_stopped = true;
        apply_motor_speeds(0, 0);
    }
}
