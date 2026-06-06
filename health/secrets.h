#pragma once

#include <string>
#include <unordered_map>

static const std::unordered_map<std::string, std::string> WIFI_NETWORKS = {\
  {"Tuan Thu","tuanthu123"},
  {"Gnas", "plmoknijb"},
  {"Khong biet5G", "216nguyendinhtuu"}
};

static constexpr const char* MQTT_SERVER = "11060dbd13b54fc988ae8f9bfc43c089.s1.eu.hivemq.cloud";
static constexpr int MQTT_PORT = 8883;
static constexpr const char* MQTT_USER = "heart-rate";
static constexpr const char* MQTT_PASS = "aB123456";

static constexpr const char* TOPIC_SENSOR = "sensor/data";
static constexpr const char* TOPIC_ALERT = "sensor/alert";
static constexpr const char* TOPIC_FALL_RAW = "sensor/fall_raw";
static constexpr const char* TOPIC_COMMAND = "device/command";

