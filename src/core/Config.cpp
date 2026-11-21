#include "core/Config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pai {
namespace {

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

}  // namespace

KeyValueFile KeyValueFile::parse(std::istream& in) {
    KeyValueFile file;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line.substr(0, line.find('#')));
        const auto eq = line.find('=');
        if (line.empty() || eq == std::string::npos) continue;
        file.values_[lower(trim(line.substr(0, eq)))] = trim(line.substr(eq + 1));
    }
    return file;
}

KeyValueFile KeyValueFile::load(const std::string& path) {
    std::ifstream in(path);
    return in ? parse(in) : KeyValueFile{};
}

std::string KeyValueFile::getString(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(lower(key));
    return it == values_.end() ? fallback : it->second;
}

double KeyValueFile::getDouble(const std::string& key, double fallback) const {
    const auto it = values_.find(lower(key));
    if (it == values_.end()) return fallback;
    try {
        return std::stod(it->second);
    } catch (const std::exception&) {
        throw std::runtime_error("config: '" + key + "' is not a number: " + it->second);
    }
}

int KeyValueFile::getInt(const std::string& key, int fallback) const {
    return static_cast<int>(getDouble(key, fallback));
}

bool KeyValueFile::getBool(const std::string& key, bool fallback) const {
    const auto it = values_.find(lower(key));
    if (it == values_.end()) return fallback;
    const std::string v = lower(it->second);
    if (v == "true" || v == "yes" || v == "on" || v == "1") return true;
    if (v == "false" || v == "no" || v == "off" || v == "0") return false;
    throw std::runtime_error("config: '" + key + "' must be true or false: " + it->second);
}

GameConfig GameConfig::from(const KeyValueFile& f) {
    GameConfig c;
    const std::string mode = lower(f.getString("mode", "demo"));
    if (mode == "demo")
        c.mode = Mode::Demo;
    else if (mode == "compare")
        c.mode = Mode::Compare;
    else if (mode == "train")
        c.mode = Mode::Train;
    else
        throw std::runtime_error("config: mode must be train, demo or compare, not " + mode);

    const std::string view = lower(f.getString("camera_view", "chase"));
    if (view == "chase")
        c.cameraView = CameraView::Chase;
    else if (view == "agent")
        c.cameraView = CameraView::Agent;
    else
        throw std::runtime_error("config: camera_view must be chase or agent, not " + view);

    const std::string demo = lower(f.getString("demonstrator", "keyboard"));
    if (demo == "keyboard")
        c.demonstrator = Demonstrator::Keyboard;
    else if (demo == "autopilot")
        c.demonstrator = Demonstrator::Autopilot;
    else
        throw std::runtime_error("config: demonstrator must be keyboard or autopilot, not " + demo);

    c.daggerShare = std::clamp(f.getDouble("dagger_share", c.daggerShare), 0.0, 1.0);
    c.mirror = f.getBool("mirror", c.mirror);
    c.collisions = f.getBool("collisions", c.collisions);
    c.neuralAgents = std::clamp(f.getInt("neural_agents", c.neuralAgents), 0, 8);
    c.scriptedAgents = std::clamp(f.getInt("scripted_agents", c.scriptedAgents), 0, 64);
    c.coins = std::clamp(f.getInt("coins", c.coins), 0, 500);
    c.respawnCoins = f.getBool("respawn_coins", c.respawnCoins);
    c.roundSeconds = std::max(1.0, f.getDouble("round_seconds", c.roundSeconds));
    c.sensorWidth = std::clamp(f.getInt("sensor_width", c.sensorWidth), 1, 64);
    c.sensorHeight = std::clamp(f.getInt("sensor_height", c.sensorHeight), 1, 64);
    c.hiddenUnits = std::clamp(f.getInt("hidden_units", c.hiddenUnits), 1, 1024);
    c.decisionInterval = std::max(0.01, f.getDouble("decision_interval", c.decisionInterval));
    c.learningRate = f.getDouble("learning_rate", c.learningRate);
    c.batchSize = std::max(1, f.getInt("batch_size", c.batchSize));
    c.replayCapacity = std::max(c.batchSize, f.getInt("replay_capacity", c.replayCapacity));
    c.policyFile = f.getString("policy_file", c.policyFile);
    c.startupScript = f.getString("startup_script", c.startupScript);
    c.autosaveSeconds = f.getDouble("autosave_seconds", c.autosaveSeconds);
    c.seed = static_cast<unsigned>(f.getInt("seed", static_cast<int>(c.seed)));
    if (c.cameraView == CameraView::Agent && c.neuralAgents == 0) c.cameraView = CameraView::Chase;
    return c;
}

}  // namespace pai
