-- Runs once at start-up, after ScriptManager.lua. Open the console with ` and type help().

-- Example: a script that logs the first cube's place on the track every five seconds.
local logger = Script.create()
local elapsed = 0
logger.Think = function(self, dt)
    elapsed = elapsed + dt
    if elapsed >= 5 then
        elapsed = 0
        local u, v = getTrackPosition(cars()[1])
        debugOutput(string.format(cars()[1] .. "  u = %.2f  v = %.2f", u, v))
    end
end
logger.active = false   -- set to true, or type  addScript(...)  in the console, to enable
addScript(logger, "logger")
