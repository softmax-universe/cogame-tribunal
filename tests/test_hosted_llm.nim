## Hosted calls must reach the native sidecar without provider credentials.
include "../src/tribunal/llm"

block:
  putEnv("COWORLD_LLM_ENDPOINT", "http://127.0.0.1:9100/")
  putEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
  putEnv("ANTHROPIC_API_KEY", "local-key-must-not-be-used")
  let client = newLlmClient(GameConfig())
  doAssert client.transport == ltSidecar
  for slot in 0 .. 1:
    let request = client.requestFor("rules", "private view", slot)
    doAssert request.url == "http://127.0.0.1:9100/v1/messages"
    doAssert request.headers["X-Coworld-Player-Slot"] == $slot
    doAssert request.headers["x-api-key"].len == 0
    let body = parseJson(request.body)
    doAssert body["model"].getStr() == "anthropic/claude-haiku-4.5"
    doAssert not body.hasKey("anthropic_version")
    doAssert not body.hasKey("output_config")
  echo "hosted sidecar routing and seat attribution passed"
