-- Unattended measurement: start a round at once, and when it ends print every
-- cube's score and quit. Use with  startup_script = scripts/evaluate.lua
Start()
local watcher = Script.create()
watcher.Think = function(self, dt)
    if roundOver() then
        local line = "result"
        for _, name in ipairs(cars()) do line = line .. "  " .. name .. "=" .. score(name) end
        debugOutput(line .. "  coins_left=" .. coinsLeft())
        Quit()
    end
end
addScript(watcher, "evaluate")
