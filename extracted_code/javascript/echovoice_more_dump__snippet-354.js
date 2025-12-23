const odi = require("../src/services/emotionIntegrators");
const res = await odi.runReflectionAndMetaUpdate(node, [{optimalValue:10, actualValue:5, emotions: [1,1,1]}]);
