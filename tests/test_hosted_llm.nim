import std/[json, os]
include "../src/tribunal/llm"

block:
  putEnv("COWORLD_LLM_ENDPOINT", "http://127.0.0.1:9100/")
  putEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
  putEnv("ANTHROPIC_API_KEY", "local-key-must-not-be-used")
  let client = newLlmClient(GameConfig())
  doAssert client.transport == ltSidecar
  for slot in 0 .. 1:
    let overrideModel = if slot == 1: "openai/gpt-4o-mini" else: ""
    let request = client.requestFor("rules", "private view", slot, overrideModel)
    doAssert request.url == "http://127.0.0.1:9100/v1/chat/completions"
    doAssert request.headers["X-Coworld-Player-Slot"] == $slot
    doAssert request.headers["x-api-key"].len == 0
    let body = parseJson(request.body)
    let expectedModel =
      if slot == 1: "openai/gpt-4o-mini"
      else: "anthropic/claude-haiku-4.5"
    doAssert body["model"].getStr() == expectedModel
    doAssert body["messages"].len == 2
    doAssert body["messages"][0]["role"].getStr() == "system"
    doAssert body["messages"][1]["role"].getStr() == "user"
    doAssert not body.hasKey("anthropic_version")
    doAssert not body.hasKey("output_config")
  let okResp = Response(
    code: 200,
    body: """{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"{\"vote\":\"guilty\"}"}}]}"""
  )
  doAssert client.textOf(okResp, "", "http://127.0.0.1:9100/v1/chat/completions") == """{"vote":"guilty"}"""
  var sawTransport = false
  try:
    discard client.textOf(Response(code: 503, body: "upstream down"), "", "http://127.0.0.1:9100/v1/chat/completions")
  except LlmTransportError:
    sawTransport = true
  doAssert sawTransport
  echo "hosted sidecar routing, per-seat model selection, and transport classification passed"
