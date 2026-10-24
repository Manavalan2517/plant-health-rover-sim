/**
 * @file protocol.h
 * @brief Wire protocol definitions for ESP32 Plant-Health Rover real-time controller.
 * Matches Python SerialCodec and CRC-16/CCITT-FALSE byte-for-byte.
 */
#ifndef ROVER_PROTOCOL_H
#define ROVER_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SERIAL_FRAME_START   0xAA
#define SERIAL_MAX_PAYLOAD   250

typedef enum {
    SERIAL_TYPE_COMMAND   = 0x01,
    SERIAL_TYPE_TELEMETRY = 0x02,
    SERIAL_TYPE_HEARTBEAT = 0x03,
    SERIAL_TYPE_STOP      = 0x04,
    SERIAL_TYPE_ACK       = 0x7F,
    SERIAL_TYPE_FAULT     = 0x80
} SerialFrameType;

/**
 * @brief Drive command payload (4 bytes, Network Byte Order / Big Endian)
 */
typedef struct __attribute__((packed)) {
    int16_t speed_pct;  /**< Forward/Reverse velocity percentage: -100 to +100 */
    int16_t steer_pct;  /**< Turning differential percentage: -100 to +100 */
} DriveCommandPayload;

/**
 * @brief Telemetry payload (10 bytes, Network Byte Order / Big Endian)
 */
typedef struct __attribute__((packed)) {
    int32_t left_encoder_ticks;
    int32_t right_encoder_ticks;
    uint8_t motor_enabled;
    uint8_t fault_flags;
} MotorTelemetryPayload;

/**
 * @brief Calculates CRC-16/CCITT-FALSE over data.
 * Polynomial: 0x1021, Initial: 0xFFFF, RefIn: False, RefOut: False, XorOut: 0x0000
 */
static inline uint16_t crc16_ccitt_false(const uint8_t *data, size_t length) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; ++i) {
        crc ^= ((uint16_t)data[i]) << 8;
        for (int b = 0; b < 8; ++b) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}

#ifdef __cplusplus
}
#endif

#endif // ROVER_PROTOCOL_H
