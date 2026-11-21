#pragma once

#include <istream>
#include <map>
#include <string>

namespace pai {

/// Minimal "key = value" file reader ('#' starts a comment). Kept free of
/// engine types so the simulation can be configured and tested without Ogre.
class KeyValueFile {
public:
    static KeyValueFile parse(std::istream& in);
    static KeyValueFile load(const std::string& path);  // empty if the file is missing

    std::string getString(const std::string& key, const std::string& fallback) const;
    double getDouble(const std::string& key, double fallback) const;
    int getInt(const std::string& key, int fallback) const;
    bool getBool(const std::string& key, bool fallback) const;

private:
    std::map<std::string, std::string> values_;
};

enum class Mode {
    Train,    ///< you (or the autopilot) drive the cube while the network learns from its camera
    Demo,     ///< the trained bot drives on its own and collects the coins
    Compare,  ///< as Demo, plus a cube you drive, to compare your score with the bot's
};

enum class Demonstrator {
    Keyboard,   ///< you drive with the keys
    Autopilot,  ///< a scripted expert presses the keys
};

enum class CameraView {
    Chase,  ///< behind the followed cube (yours if you drive one, otherwise the bot's)
    Agent,  ///< through the bot's own camera
};

/// Every setting that used to be hard-coded in one of the 2016 builds.
struct GameConfig {
    Mode mode = Mode::Demo;
    CameraView cameraView = CameraView::Chase;
    bool mirror = true;  ///< rear-view inset
    bool collisions = true;

    int neuralAgents = 1;    ///< bots driven by the policy network
    int scriptedAgents = 0;  ///< lane-keeping obstacle cubes at a constant speed
    int coins = 74;
    bool respawnCoins = false;  ///< false: a collected coin is gone, as in 2016

    double roundSeconds = 90.0;

    // Perception and policy: sensorWidth x sensorHeight grey pixels -> hidden -> 4 actions.
    // The 2016 builds used 3x3 -> 6; their experiments also tried 8x8 up to 80x80.
    // 20x20 is the default because a coin covers less than one pixel of a 3x3 image.
    int sensorWidth = 20;
    int sensorHeight = 20;
    int hiddenUnits = 32;
    double decisionInterval = 0.1;  ///< seconds between network decisions
    double learningRate = 0.05;
    int batchSize = 5;
    int replayCapacity = 2048;
    Demonstrator demonstrator = Demonstrator::Keyboard;  ///< who drives your cube in training mode
    /// With the autopilot: share of decisions where the network drives while the
    /// autopilot still labels (DAgger), so the network also learns to recover.
    double daggerShare = 0.0;
    std::string policyFile = "models/policy.txt";
    std::string startupScript = "scripts/startup.lua";
    double autosaveSeconds = 15.0;

    unsigned seed = 2016;

    static GameConfig from(const KeyValueFile& file);
    int sensorPixels() const { return sensorWidth * sensorHeight; }
};

}  // namespace pai
