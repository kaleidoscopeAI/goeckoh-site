const wmDecayMod = clamp( sigmoid(projectVec(s, DEFAULT_P.memory_write) ) , 0, 1);
