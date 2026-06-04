#pragma once

#include <string>
#include <unordered_map>

static const std::unordered_map<std::string, std::string> WIFI_NETWORKS = {
  {"your-wifi-ssid", "your-wifi-password"},
};

static constexpr const char* MQTT_SERVER = "your-mqtt-broker.example.com";
static constexpr int MQTT_PORT = 8883;
static constexpr const char* MQTT_USER = "your-mqtt-username";
static constexpr const char* MQTT_PASS = "your-mqtt-password";

static constexpr const char* TOPIC_SENSOR = "sensor/data";
static constexpr const char* TOPIC_ALERT = "sensor/alert";
static constexpr const char* TOPIC_FALL_RAW = "sensor/fall_raw";
static constexpr const char* TOPIC_COMMAND = "device/command";
