-- A small script system. The game calls ScriptManagerThink(dt) every frame
-- (in the menu too); it calls Think on each active script.
--
--   local s = Script.create()
--   s.Think = function(self, dt) ... end
--   addScript(s, "name")

Script = {}
Script.__index = Script

function Script.create()
    return setmetatable({ active = true }, Script)
end

function Script:Active() return self.active end
function Script:Think(dt) end

local scripts = {}

function addScript(script, name)
    scripts[name or (#scripts + 1)] = script
end

function removeScript(name)
    scripts[name] = nil
end

function ScriptManagerThink(dt)
    for _, s in pairs(scripts) do
        if s:Active() then s:Think(dt) end
    end
end
