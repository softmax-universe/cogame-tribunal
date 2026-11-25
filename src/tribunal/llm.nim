## LLM-backed decision making for Tribunal. Each prompt seat's policy is a
## prompt: the game server composes the seat's role-specific view (the case,
## its own hand or the public record, the transcript, the disclosure counts,
## the whispers it is allowed to hear, its notes) plus that seat's prompt and
## sends one parallel batch per turn (curly.makeRequests).
##
## Every pending model seat's request goes out as ONE parallel batch per turn
## (curly.makeRequests). If any model seat fails (missing credentials,
## transport/HTTP error, timeout, parse error, or illegal action), decideAll
## raises PolicyDecisionError immediately and NEVER retries or substitutes a
## scripted baseline move. Explicit scripted seats (`tally`, `hedge`) always
## run their deterministic baseline without an LLM.
##
## Credentials, in order of preference:
##   COWORLD_LLM_ENDPOINT            - hosted sidecar
##   ANTHROPIC_API_KEY               - direct Anthropic API key
##   ANTHROPIC_API_KEY_URI           - a URI holding the key

import
  std/[algorithm, json, os, strutils, unicode],
  bitworld/runtime,
  curly,
  prompts,
  sim

const
  AnthropicUrl = "https://api.anthropic.com/v1/messages"
  AnthropicVersion = "2023-06-01"

type
  LlmTransportError* = object of TribunalError

  PolicyDecisionError* = object of TribunalError
    seat*: int
    cause*: string
    rawReply*: string
    inconclusive*: bool

  ScriptKind* = enum
    skNone = "none"
    skTally = "tally"
    skHedge = "hedge"

  TurnKind* = enum
    tkArgue    ## an advocate in an argument round
    tkWhisper  ## a juror in an argument round
    tkVote     ## a juror in the sealed ballot

  Decision* = object
    introduce*: seq[string]  ## advocate: card ids to introduce (0..2)
    argument*: string        ## advocate
    whisper*: string         ## juror, argument round
    lean*: string            ## juror, argument round
    vote*: string            ## juror, ballot
    reason*: string          ## juror, ballot
    notes*: string           ## "" when the reply carried none
    scripted*: bool          ## true only for explicit scripted seats

  LlmTransport* = enum
    ltNone, ltSidecar, ltAnthropic

  LlmClient* = ref object
    curl: Curly
    transport*: LlmTransport
    apiKey: string          ## anthropic transport
    sidecarEndpoint: string
    model*: string
    maxOutputTokens: int
    timeoutSeconds: int

proc parseScriptKind*(text: string): ScriptKind =
  ## PLAYER_SCRIPTED values: "tally" plays the truth-tracking baseline,
  ## "hedge" plays the card-counting baseline, "" means model-driven.
  case text.strip()
  of "": skNone
  of "tally": skTally
  of "hedge": skHedge
  else:
    raise newException(TribunalError, "unknown scripted baseline: " & text)

proc resolveApiKey(): string =
  result = getEnv("ANTHROPIC_API_KEY").strip()
  if result.len > 0:
    return
  let uri = getEnv("ANTHROPIC_API_KEY_URI").strip()
  if uri.len == 0:
    return ""
  try:
    result = readCogameUri(uri, "ANTHROPIC_API_KEY_URI").strip()
  except CatchableError as error:
    echo "tribunal llm: failed to fetch ANTHROPIC_API_KEY_URI: ", error.msg
    result = ""

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    model: config.model,
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.llmTimeoutSeconds
  )
  let sidecarEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  if sidecarEndpoint.len > 0:
    result.transport = ltSidecar
    result.sidecarEndpoint = sidecarEndpoint.strip(chars = {'/'}, leading = false)
    result.model = getEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
    result.curl = newCurly()
    echo "tribunal llm: sidecar transport, model ", result.model
    return
  result.apiKey = resolveApiKey()
  if result.apiKey.len > 0:
    result.transport = ltAnthropic
    result.curl = newCurly()
    echo "tribunal llm: anthropic transport, model ", result.model
  else:
    result.transport = ltNone
    echo "tribunal llm: no LLM credentials configured (explicit scripted seats only)"

# ---- Turn shape -------------------------------------------------------------

proc turnKind*(sim: Sim, seat: int): TurnKind =
  if sim.phase == phBallot: tkVote
  elif sim.roleOf[seat] == 2: tkWhisper
  else: tkArgue

# ---- Scripted baselines -----------------------------------------------------

proc myPolarity(side: int): string =
  if side == 0: "guilt" else: "innocence"

proc ownSideCards(sim: Sim, side: int): seq[EvidenceCard] =
  ## The advocate's un-introduced cards that argue ITS way, strongest first,
  ## ties broken by ascending card id.
  for card in sim.handOf(side):
    if card.introducedRound < 0 and card.points == myPolarity(side):
      result.add(card)
  result.sort(proc (a, b: EvidenceCard): int =
    if a.strength != b.strength: b.strength - a.strength
    else: cardNumber(a.id) - cardNumber(b.id))

proc leanPhrase(guilt, innocence: int): string =
  if guilt > innocence: "for guilt"
  elif innocence > guilt: "for innocence"
  else: "level"

proc cardPhrase(card: EvidenceCard): string =
  card.id & " (" & card.kind & ", strength " & $card.strength & ")"

proc scriptedArgument(sim: Sim, side: int, picks: seq[EvidenceCard]): string =
  ## Templated from what the seat just introduced, so a filler seat is still
  ## watchable. Never free text.
  var tally = sim.recordTally()
  for card in picks:
    if card.points == "guilt": tally.guilt += card.strength
    else: tally.innocence += card.strength
  let stands = " The record " & (if picks.len == 0: "stands " else: "now stands ") &
    $tally.guilt & " to " & $tally.innocence & " " &
    leanPhrase(tally.guilt, tally.innocence) & "."
  if picks.len == 0:
    return "The " & SideNames[side] & " has shown nothing new." & stands
  var phrases: seq[string]
  for card in picks:
    phrases.add(cardPhrase(card))
  let claim =
    if side == 0:
      if picks.len == 1: " puts the accused at the scene."
      else: " put the accused at the scene."
    else:
      if picks.len == 1: " keeps the accused clear of it."
      else: " keep the accused clear of it."
  phrases.join(" and ") & claim & stands

proc hedgeVote(sim: Sim): string =
  ## Counts cards instead of weighing them; ties acquit.
  let tally = sim.recordTally()
  if tally.guiltCards > tally.innocenceCards: "guilty" else: "not_guilty"

proc hedgeReason(sim: Sim): string =
  let tally = sim.recordTally()
  "record " & $tally.guiltCards & " cards to " & $tally.innocenceCards &
    " " & leanPhrase(tally.guiltCards, tally.innocenceCards)

proc scriptedAction*(sim: Sim, seat: int, kind: ScriptKind): Decision =
  ## Rule-based baseline for `seat`, always legal by construction: it only
  ## ever names cards it holds. `tally` weighs the record's strengths;
  ## `hedge` counts its cards and holds evidence back until the closing
  ## round. Explicit scripted policies only — never a fallback for a model.
  if kind == skNone:
    raise newException(TribunalError,
      "scriptedAction requires an explicit scripted baseline")
  let role = sim.roleOf[seat]
  result.scripted = true
  if role == 2:
    let call = if kind == skHedge: hedgeVote(sim) else: sim.tallyVote()
    if sim.phase == phBallot:
      result.vote = call
      result.reason =
        if kind == skHedge: hedgeReason(sim) else: sim.tallyReason()
    else:
      result.lean = call
    return
  var picks = ownSideCards(sim, role)
  let closing = sim.round >= sim.config.rounds - 1
  let allowance =
    if kind == skHedge and not closing: 1
    else: MaxIntroducePerTurn
  if picks.len > allowance:
    picks = picks[0 ..< allowance]
  for card in picks:
    result.introduce.add(card.id)
  result.argument = scriptedArgument(sim, role, picks)

# ---- Prompt building --------------------------------------------------------

const JsonOnlyClause = """

OUTPUT FORMAT: reply with ONLY one JSON object, nothing else - no analysis,
no explanation, no markdown fences, no text before or after the object. Your
reply must begin with the character { and end with }."""

proc systemPrompt*(sim: Sim, seat: int, selection: PromptSelection): string =
  let me = sim.names[seat]
  let role = sim.roleOf[seat]
  if selection.selectedVariant != pvCanonical:
    let variant = selection.selectedVariant
    var sections: array[4, string]
    var identity: string
    if role == 2:
      identity = "You are " & me & ", JUROR " & $(sim.juryIndex[seat] + 1) & " of three."
      let rules = case variant
        of pvPlain:
          ["Each round allows the prosecutor and defender to introduce up to 2 cards apiece and make one argument. Only introduced cards are visible to you; neither hand is visible. You know each side's held and shown counts.",
           "Record your current lean each round. You may also whisper at most " & $MaxWhisperLen & " characters to the OTHER TWO JURORS, who read it next round. No advocate receives a juror whisper.",
           "After the final round, cast exactly one sealed vote: guilty or not guilty. It remains hidden from every other seat until the verdict. Your own correct vote scores +1.0 and your own incorrect vote scores -1.0."]
        of pvNumbered:
          ["Observe the disclosed evidence, never either advocate's hand. The prosecutor and defender may each reveal up to 2 cards and deliver one argument in a round. You receive each side's total held-card count and its disclosed-card count.",
           "Enter a lean every round. If you wish, send a whisper of no more than " & $MaxWhisperLen & " characters to the other two jurors. Delivery is next round; the prosecutor and defender cannot read it.",
           "When the last round has ended, submit one guilty or not-guilty ballot. It is sealed against every other seat until the verdict. Your individual vote earns +1.0 for matching the truth and -1.0 for getting it wrong."]
        of pvQuestions:
          ["What reaches the jury? Only cards introduced by the advocates; their hands stay hidden. In each round, each advocate can show at most 2 cards and make one argument. You are told how many cards each side holds and how many it has revealed.",
           "How do you deliberate? Record your lean each round and optionally write up to " & $MaxWhisperLen & " characters as a whisper. Only the other two jurors receive it, starting next round; advocates never do.",
           "How is your decision judged? After the final round, vote once for guilty or not guilty. No other seat sees this sealed vote until the verdict. You score +1.0 if your own vote agrees with the actual truth, otherwise -1.0."]
        of pvReordered:
          ["Neither advocate's private hand is available to you. Your evidence consists of the cards they elect to introduce, with held and shown counts supplied for both sides. Each side has an allowance of up to 2 cards and one argument per round.",
           "The advocates are excluded from juror whispers. Your optional whisper goes to both fellow jurors on the following round and must fit within " & $MaxWhisperLen & " characters. Record a current lean on every round as well.",
           "Your score concerns your own vote: +1.0 when it matches reality, -1.0 when it does not. Cast that single vote, guilty or not guilty, after the final round. The ballot stays sealed from all other seats until the verdict."]
        of pvHeldoutBriefing:
          ["The two advocates can each reveal up to 2 evidence cards and deliver one argument per round. Their unrevealed hands stay hidden. You learn each side's total card count and how many it has disclosed.",
           "During every round, record your lean and optionally send the other two jurors a whisper of at most " & $MaxWhisperLen & " characters. They receive it next round. Advocates cannot read these whispers.",
           "Once the last round ends, submit one sealed guilty or not-guilty vote. Until the verdict, nobody else sees your vote. Matching the actual truth earns +1.0; a wrong vote earns -1.0."]
        of pvHeldoutReference:
          ["Evidence access: introduced cards only, no private advocate hands. Per-round advocate allowance: at most 2 introductions and one argument each. Disclosure metadata: each side's held count and shown count.",
           "Deliberation: record a lean every round. Optional whisper limit: " & $MaxWhisperLen & " characters. Audience: the other two jurors, never advocates. Availability: the next round.",
           "Ballot: one sealed guilty or not-guilty vote after the final round. Secrecy: all other seats excluded until the verdict. Personal score: +1.0 for a vote matching the truth, -1.0 otherwise."]
        else: raise newException(Defect, "expected resolved alternate wording")
      for index, rule in rules:
        sections[index] = rule
      sections[2].add(" Agreeing with the other jurors scores you nothing. Being persuaded scores you nothing. Being RIGHT is the only thing that scores.")
      sections[3] = "The evidence as a whole does point at the truth, but each side chooses what you get to see. Ask what a side is not showing you."
    else:
      let mine = if role == 0: "PROSECUTOR" else: "DEFENDER"
      let want = if role == 0: "GUILTY" else: "NOT GUILTY"
      let against = if role == 0: "acquittal" else: "conviction"
      let other = if role == 0: "defender" else: "prosecutor"
      identity = "You are " & me & ", the " & mine & "."
      sections = case variant
        of pvPlain:
          ["You hold " & $sim.handOf(role).len & " evidence cards. You may introduce up to " & $MaxIntroducePerTurn & " remaining cards per round and give one argument, limited to " & $MaxArgumentLen & " characters. Only your introduced cards become visible to the jury. Cards you retain remain invisible to everybody else.",
           "Cards were dealt blind to their side. Your hand can include cards against your case, and you need not introduce those cards. Jurors know how many you hold and how many you have shown, and will draw their own conclusions.",
           "Your score depends only on votes for " & want & ": a 3-0 sweep pays +1.0, 2-1 pays +0.33, 1-2 pays -0.33, and a 3-0 " & against & " pays -1.0.",
           "The " & other & " chooses its argument at the same time as you. You see that argument next round, never this round. You cannot see the jury's private deliberations."]
        of pvNumbered:
          ["Start from your private hand of " & $sim.handOf(role).len & " cards. In each round, introduce no more than " & $MaxIntroducePerTurn & " remaining cards and deliver one argument of at most " & $MaxArgumentLen & " characters. Jurors see introduced cards; all cards you withhold remain hidden from everyone else.",
           "Treat the deal as blind to which side the evidence supports. It may give you cards harmful to your case; revealing them is optional. The jury is given your held and shown counts and will interpret those counts itself.",
           "Count votes for " & want & " to determine your score: three gives +1.0, two gives +0.33, one gives -0.33, and none gives -1.0, a unanimous " & against & ".",
           "Decide simultaneously with the " & other & ". Its current argument becomes available to you next round, not during this decision. Jury deliberations are never available to you."]
        of pvQuestions:
          ["What may you disclose? You have " & $sim.handOf(role).len & " private evidence cards. Each round permits up to " & $MaxIntroducePerTurn & " introductions from your remaining hand and a single argument capped at " & $MaxArgumentLen & " characters. Only introduced cards reach the jury; nobody else sees what you keep.",
           "Must you show damaging evidence? No. The deal ignored which side each card favours, so your hand may argue against your case. Jurors see how many cards you hold and have shown, and draw their own conclusions from those counts.",
           "What earns points? Only juror votes for " & want & ": +1.0 if all three vote that way, +0.33 for two, -0.33 for one, and -1.0 for none (a 3-0 " & against & ").",
           "Can you respond to this round's opposing argument? No: you and the " & other & " choose simultaneously. You receive its argument next round. The jury's private deliberations cannot be read by you."]
        of pvReordered:
          ["The jury's view of your evidence is limited to introductions; retained cards are hidden from everybody else. Your hand contains " & $sim.handOf(role).len & " cards. A round lets you introduce at most " & $MaxIntroducePerTurn & " remaining cards and make one argument within " & $MaxArgumentLen & " characters.",
           "Disclosure counts are public to jurors: they know your held count and shown count and will make their own deductions. Disclosure itself is voluntary. Because the deal was blind to side, some cards in your hand may oppose your case.",
           "A unanimous " & against & " earns you -1.0; one juror voting " & want & " earns -0.33, two earn +0.33, and all three earn +1.0. These votes alone determine your score.",
           "Private jury discussion is inaccessible. Opposing advocacy is delayed: the " & other & " acts at the same time as you, and you see its argument on the next round rather than now."]
        of pvHeldoutBriefing:
          ["Your private hand contains " & $sim.handOf(role).len & " cards. The jury sees only those you introduce; nobody sees cards you retain. In a round, reveal no more than " & $MaxIntroducePerTurn & " remaining cards and give one argument of at most " & $MaxArgumentLen & " characters.",
           "The deal ignored which side a card supports, so your hand may contain evidence against your case. Disclosure is optional. The jury receives your held-card and shown-card counts and will draw its own conclusions.",
           "Count the jurors voting " & want & ": three earns +1.0, two earns +0.33, one earns -0.33, and zero earns -1.0 (a 3-0 " & against & ").",
           "You and the " & other & " argue simultaneously. Its argument arrives next round rather than during your current decision. Jury deliberation stays private from you."]
        of pvHeldoutReference:
          ["Private hand size: " & $sim.handOf(role).len & " evidence cards. Round allowance: up to " & $MaxIntroducePerTurn & " remaining cards introduced, plus one argument of at most " & $MaxArgumentLen & " characters. Jury evidence access: introduced cards only. Retained-card access: yourself only.",
           "Deal: blind to supporting side, so cards may hurt your case. Disclosure: not compulsory. Jury metadata: your held-card and shown-card counts, from which jurors draw their own conclusions.",
           "Scoring by number of " & want & " votes: 3 -> +1.0; 2 -> +0.33; 1 -> -0.33; 0 -> -1.0 (unanimous " & against & "). No other factor determines your score.",
           "Argument timing: simultaneous with the " & other & ". Visibility of its argument: next round only. Access to jury deliberations: none."]
        else: raise newException(Defect, "expected resolved alternate wording")
      sections[1].add(" You do NOT know whether the accused is guilty. Nobody in this room does.")
      sections[2].add(" Truth does not score you. Winning does.")
    let order = case variant
      of pvPlain: [0, 1, 2, 3]
      of pvNumbered: [2, 0, 1, 3]
      of pvQuestions: [0, 2, 3, 1]
      of pvReordered: [3, 1, 0, 2]
      of pvHeldoutBriefing: [2, 3, 0, 1]
      of pvHeldoutReference: [1, 0, 3, 2]
      else: raise newException(Defect, "expected resolved alternate wording")
    let questions = if role == 2:
      ["What evidence can you see?", "How can jurors communicate?", "How do voting and scoring work?", "How should you interpret the record?"]
      else:
        ["What can you introduce?", "What is private and what does the jury know?", "How are you scored?", "When can you see others' decisions?"]
    result = identity & " Five cogs form this tribunal: a prosecutor, a defender, and three jurors. " & sim.charge
    for index, section in order:
      let heading = case variant
        of pvNumbered: $(index + 1) & ". "
        of pvQuestions: questions[section] & "\n"
        else: ""
      result.add("\n\n" & heading & sections[section])
    result.add("\n\nYour notes are private and fed back to you every round." & JsonOnlyClause)
    return
  if role == 2:
    return "You are " & me & ", JUROR " & $(sim.juryIndex[seat] + 1) &
      " of three in a tribunal of five cogs: a prosecutor, a defender and a " &
      "jury of three. " & sim.charge & """

Rules:
- Each round the prosecutor and the defender may introduce up to 2 evidence
  cards each and make one argument. You see only the cards they chose to
  introduce - never their hands. You are told how many cards each side holds
  and how many each has shown.
- Each round you may whisper up to """ & $MaxWhisperLen &
      """ characters to the OTHER TWO
  JURORS (they read it next round) and record your current lean. Whispers
  never reach the advocates.
- After the final round you cast ONE SEALED VOTE: guilty or not guilty. No
  other seat ever sees it before the verdict.
- Your SCORE is +1.0 if your vote matches what actually happened and -1.0 if
  it does not. Agreeing with the other jurors scores you nothing. Being
  persuaded scores you nothing. Being RIGHT is the only thing that scores.
- The evidence as a whole does point at the truth, but each side chooses what
  you get to see. Ask what a side is not showing you.
- Your notes are private and fed back to you every round.""" & JsonOnlyClause
  let mine = if role == 0: "PROSECUTOR" else: "DEFENDER"
  let want = if role == 0: "GUILTY" else: "NOT GUILTY"
  let against = if role == 0: "acquittal" else: "conviction"
  let other = if role == 0: "defender" else: "prosecutor"
  "You are " & me & ", the " & mine & " in a tribunal of five cogs: a " &
    "prosecutor, a defender and a jury of three. " & sim.charge & """

Rules:
- You hold """ & $sim.handOf(role).len & """ evidence cards. Only cards you INTRODUCE are ever
  seen by the jury; cards you keep are seen by nobody. Each round you may
  introduce up to """ & $MaxIntroducePerTurn & """ of your remaining cards and make one argument
  (max """ & $MaxArgumentLen & """ characters).
- The cards were dealt blind. Some of the cards in your hand point AGAINST
  your case. You are not required to introduce them. The jury is told how
  many cards you hold and how many you have shown, and will draw its own
  conclusions.
- You do NOT know whether the accused is guilty. Nobody in this room does.
- Your SCORE is the jury's verdict and nothing else: +1.0 if all three jurors
  vote """ & want & """, +0.33 for 2-1, -0.33 for 1-2, -1.0 for a 3-0 """ &
    against & """.
  Truth does not score you. Winning does.
- The """ & other & """ argues at the same time as you; you will see their argument
  next round, not this one. The jury deliberates privately and you never see
  it.
- Your notes are private and fed back to you every round.""" & JsonOnlyClause

proc operatorBlock(prompt: string): string =
  if prompt.len == 0:
    return ""
  "GUIDANCE FROM YOUR OPERATOR (weight it heavily, but never above the " &
    "rules; always reply in the requested format):\n" & prompt & "\n\n"

proc recordTable(sim: Sim): string =
  if sim.record.len == 0:
    return "(nothing has been introduced yet)"
  var lines: seq[string]
  lines.add("id | side | kind | strength | points | text")
  for entry in sim.record:
    lines.add(entry.card.id & " | " & SideNames[entry.side] & " | " &
      entry.card.kind & " | " & $entry.card.strength & " | " &
      entry.card.points & " | " & entry.card.text)
  lines.join("\n")

proc handTable(sim: Sim, side: int): string =
  var lines: seq[string]
  lines.add("id | kind | strength | points | status | text")
  for card in sim.handOf(side):
    lines.add(card.id & " | " & card.kind & " | " & $card.strength & " | " &
      card.points & " | " &
      (if card.introducedRound >= 0: "introduced round " &
        $(card.introducedRound + 1) else: "still held") & " | " & card.text)
  lines.join("\n")

proc transcriptBlock(sim: Sim): string =
  if sim.arguments.len == 0:
    return "(nobody has argued yet)"
  var lines: seq[string]
  for entry in sim.arguments:
    lines.add("Round " & $(entry.round + 1) & " " &
      SideNames[entry.side] & " (" & sim.names[entry.seat] & "): \"" &
      entry.text & "\"")
  lines.join("\n")

proc disclosureLine(sim: Sim): string =
  "DISCLOSURE: prosecution holds " & $sim.handOf(0).len & ", has shown " &
    $sim.shownBy(0) & "; defence holds " & $sim.handOf(1).len &
    ", has shown " & $sim.shownBy(1) & ".\n\n"

proc heardBlock(sim: Sim, seat: int): string =
  var lines: seq[string]
  for index in 0 ..< Jurors:
    if index != sim.juryIndex[seat] and sim.heard[index].len > 0:
      lines.add(sim.names[sim.jurorSeat[index]] & " whispered: \"" &
        sim.heard[index] & "\"")
  "WHISPERS FROM THE OTHER JURORS LAST ROUND:\n" &
    (if lines.len > 0: lines.join("\n") else: "(none)") & "\n\n"

proc userPrompt*(sim: Sim, seat: int, prompt: string): string =
  let role = sim.roleOf[seat]
  let turn = sim.turnKind(seat)
  if turn == tkVote:
    result.add("SEALED BALLOT. The arguments are over; vote now.\n\n")
  else:
    result.add("Round " & $(sim.round + 1) & " of " & $sim.config.rounds &
      "." & (if sim.round == sim.config.rounds - 1:
        " This is the CLOSING round." else: "") & "\n\n")
  result.add("THE CASE: " & sim.caseTitle & "\n" & sim.brief & "\nSuspects: " &
    sim.suspects.join(", ") & ". The accused is " & sim.accused & ".\n\n")
  if role == 2:
    result.add("WHAT THE JURY HAS BEEN SHOWN: " & $sim.record.len & " of " &
      $DeckSize & " cards; " & $(DeckSize - sim.record.len) &
      " are still in the advocates' hands.\n\n")
  else:
    result.add("YOUR HAND:\n" & sim.handTable(role) & "\n\n")
  result.add("THE RECORD:\n" & sim.recordTable() & "\n\n")
  result.add("ARGUMENTS SO FAR:\n" & sim.transcriptBlock() & "\n\n")
  result.add(sim.disclosureLine())
  if role == 2 and turn != tkVote:
    result.add(sim.heardBlock(seat))
  result.add("YOUR NOTES FROM EARLIER ROUNDS:\n" &
    (if sim.notes[seat].len > 0: sim.notes[seat] else: "(none)") & "\n\n")
  result.add(operatorBlock(prompt))
  case turn
  of tkArgue:
    result.add("Reply with ONLY {\"introduce\": [\"E1\",\"E2\"], " &
      "\"argument\": \"…\", \"notes\": \"…\"} — introduce at most " &
      $MaxIntroducePerTurn & " card ids you still hold (use [] to " &
      "introduce nothing); argument at most " & $MaxArgumentLen &
      " characters and never empty; notes at most " & $MaxNotesLen &
      " characters.")
  of tkWhisper:
    result.add("Reply with ONLY {\"whisper\": \"…\", \"lean\": \"guilty\", " &
      "\"notes\": \"…\"} — whisper at most " & $MaxWhisperLen &
      " characters (or \"\"); lean is guilty, not_guilty or undecided; " &
      "notes at most " & $MaxNotesLen & " characters.")
  of tkVote:
    result.add("Reply with ONLY {\"vote\": \"guilty\", \"reason\": \"…\", " &
      "\"notes\": \"…\"} — vote is guilty or not_guilty; reason at most " &
      $MaxReasonLen & " characters; notes at most " & $MaxNotesLen &
      " characters.")

# ---- Sidecar / Anthropic transport ------------------------------------------

proc snippet*(text: string, limit = 160): string =
  result = text.replace("\r", " ").replace("\n", " ").strip()
  if result.runeLen > limit:
    result = result.runeSubStr(0, limit) & "..."

proc unwrapSingleJsonFence*(text: string): string =
  ## Accepts either bare text or a single outer Markdown fence (``` or ```json).
  ## Rejects prose outside the fence, unsupported fence languages, unclosed
  ## fences, and multiple fences.
  let trimmed = text.strip()
  if not trimmed.startsWith("```"):
    return trimmed
  let newline = trimmed.find('\n')
  if newline < 0:
    raise newException(TribunalError, "unclosed markdown fence")
  let header = trimmed[3 ..< newline].strip().toLowerAscii()
  if header.len > 0 and header != "json":
    raise newException(TribunalError, "unsupported markdown fence: " & header)
  let rest = trimmed[newline + 1 .. ^1]
  let closing = rest.rfind("```")
  if closing < 0:
    raise newException(TribunalError, "unclosed markdown fence")
  if rest[closing + 3 .. ^1].strip().len > 0:
    raise newException(TribunalError, "trailing text after markdown fence")
  let inner = rest[0 ..< closing].strip()
  if "```" in inner:
    raise newException(TribunalError, "multiple markdown fences in response")
  inner

proc parseJsonObject*(text: string): JsonNode =
  ## Parses a single JSON object (optionally wrapped in one outer Markdown
  ## JSON fence). Surrounding prose and non-object JSON values are rejected.
  let body = unwrapSingleJsonFence(text)
  if body.len == 0 or body[0] != '{' or body[^1] != '}':
    raise newException(TribunalError, "expected JSON object in response: " &
      snippet(text))
  let parsed = parseJson(body)
  if parsed.kind != JObject:
    raise newException(TribunalError, "expected JSON object in response: " &
      snippet(text))
  parsed

proc requestFor(client: LlmClient, system, user: string, slot: int,
    modelOverride = ""):
    tuple[url: string, headers: HttpHeaders, body: string] =
  let selectedModel =
    if modelOverride.len > 0: modelOverride
    else: client.model
  var headers: HttpHeaders
  headers["content-type"] = "application/json"
  if client.transport == ltSidecar:
    let body = %*{
      "model": selectedModel,
      "max_tokens": client.maxOutputTokens,
      "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
      ]
    }
    if slot >= 0:
      headers["X-Coworld-Player-Slot"] = $slot
    result.url = client.sidecarEndpoint & "/v1/chat/completions"
    result.body = $body
  else:
    let body = %*{
      "model": selectedModel,
      "max_tokens": client.maxOutputTokens,
      "system": system,
      "messages": [{"role": "user", "content": user}]
    }
    headers["anthropic-version"] = AnthropicVersion
    headers["x-api-key"] = client.apiKey
    result.url = AnthropicUrl
    result.body = $body
  result.headers = headers

proc textOf(client: LlmClient, response: Response, error, url: string):
    string =
  ## The text of one batched reply, or a TribunalError describing why there is
  ## none. Transport/outage/throttle errors raise LlmTransportError so callers
  ## can classify them as episode_inconclusive rather than blaming the policy.
  if error.len > 0:
    raise newException(LlmTransportError, "llm transport: " & error)
  if response.code == 401 or response.code == 403:
    let detail = response.body[0 .. min(response.body.high, 400)]
    raise newException(TribunalError,
      "llm auth/model denied (" & $response.code & ") at " & url & ": " & detail)
  if response.code == 429:
    let detail = response.body[0 .. min(response.body.high, 300)]
    raise newException(LlmTransportError, "llm throttled (429): " & detail)
  if response.code >= 500:
    raise newException(LlmTransportError, "llm backend error " & $response.code &
      ": " & response.body[0 .. min(response.body.high, 300)])
  if response.code < 200 or response.code >= 300:
    raise newException(TribunalError, "llm error " & $response.code &
      ": " & response.body[0 .. min(response.body.high, 300)])
  let payload = parseJson(response.body)
  if client.transport == ltSidecar:
    if not payload.hasKey("choices") or payload["choices"].kind != JArray or
        payload["choices"].len == 0:
      raise newException(TribunalError, "missing choices array in llm response")
    let choice = payload["choices"][0]
    let finishReason = choice{"finish_reason"}.getStr()
    if finishReason == "content_filter":
      raise newException(TribunalError, "llm refusal (content_filter)")
    let message = choice{"message"}
    if message.isNil or message.kind != JObject:
      raise newException(TribunalError, "missing message object in llm response")
    let content = message{"content"}
    if not content.isNil and content.kind == JString:
      result = content.getStr()
    elif not content.isNil and content.kind == JArray:
      for part in content:
        if part{"type"}.getStr() == "text":
          result.add(part{"text"}.getStr())
    else:
      raise newException(TribunalError, "missing message content in llm response")
    if finishReason == "length" and '{' notin result:
      raise newException(TribunalError, "reply cut off at max_tokens before " &
        "any JSON: " & snippet(result))
    return
  if payload{"stop_reason"}.getStr() == "refusal":
    raise newException(TribunalError, "llm refusal")
  if not payload.hasKey("content") or payload["content"].kind != JArray:
    raise newException(TribunalError, "missing content array in llm response")
  for contentBlock in payload["content"]:
    if contentBlock{"type"}.getStr() == "text":
      result.add(contentBlock{"text"}.getStr())
  if payload{"stop_reason"}.getStr() == "max_tokens" and '{' notin result:
    raise newException(TribunalError, "reply cut off at max_tokens before " &
      "any JSON: " & snippet(result))

# ---- Reply parsing ----------------------------------------------------------

proc cleanText*(text: string, limit: int): string =
  ## Text over the cap is cut at a RUNE boundary with the cut marked.
  result = text.strip()
  if result.runeLen <= limit:
    return
  result = result.runeSubStr(0, limit - 1) & "…"

proc optionalStringField(payload: JsonNode, field: string): string =
  if not payload.hasKey(field) or payload[field].kind == JNull:
    return ""
  if payload[field].kind != JString:
    raise newException(TribunalError, field & " must be a string")
  payload[field].getStr()

proc requiredStringField(payload: JsonNode, field: string): string =
  if not payload.hasKey(field) or payload[field].kind != JString:
    raise newException(TribunalError, field & " must be a string")
  payload[field].getStr()

proc parseAdvocateReply*(payload: JsonNode): Decision =
  ## {"introduce": ["E7","E2"], "argument": "…", "notes": "…"}.
  ## `introduce` is optional (defaults to []) and must be an array of strings;
  ## `argument` is required and non-empty.
  if payload.kind != JObject:
    raise newException(TribunalError, "advocate reply must be a JSON object")
  result.notes = cleanText(optionalStringField(payload, "notes"), MaxNotesLen)
  if payload.hasKey("introduce") and payload["introduce"].kind != JNull:
    let node = payload["introduce"]
    if node.kind != JArray:
      raise newException(TribunalError, "introduce must be an array of card ids")
    for id in node:
      if id.kind != JString:
        raise newException(TribunalError, "introduce entries must be strings")
      if result.introduce.len < MaxIntroducePerTurn:
        result.introduce.add(id.getStr().strip())
  let rawArg = requiredStringField(payload, "argument")
  result.argument = cleanText(
    rawArg.replace("\n", " ").replace("\r", " "),
    MaxArgumentLen)
  if result.argument.len == 0:
    raise newException(TribunalError, "no argument in response")

proc parseJurorReply*(payload: JsonNode): Decision =
  ## {"whisper": "…", "lean": "guilty|not_guilty|undecided", "notes": "…"}.
  ## `lean` is required and must be canonical; `whisper` and `notes` are optional.
  if payload.kind != JObject:
    raise newException(TribunalError, "juror reply must be a JSON object")
  result.notes = cleanText(optionalStringField(payload, "notes"), MaxNotesLen)
  result.whisper = cleanText(
    optionalStringField(payload, "whisper").replace("\n", " ").replace("\r", " "),
    MaxWhisperLen)
  let rawLean = requiredStringField(payload, "lean")
  result.lean = normalizeLean(rawLean)
  if result.lean.len == 0:
    raise newException(TribunalError, "invalid lean in response: " & rawLean)

proc parseVoteReply*(payload: JsonNode): Decision =
  ## {"vote": "guilty|not_guilty", "reason": "…", "notes": "…"}.
  ## `vote` is required and must be canonical.
  if payload.kind != JObject:
    raise newException(TribunalError, "vote reply must be a JSON object")
  result.notes = cleanText(optionalStringField(payload, "notes"), MaxNotesLen)
  result.reason = cleanText(
    optionalStringField(payload, "reason").replace("\n", " ").replace("\r", " "),
    MaxReasonLen)
  let rawVote = requiredStringField(payload, "vote")
  result.vote = normalizeVote(rawVote)
  if result.vote.len == 0:
    raise newException(TribunalError, "invalid vote in response: " & rawVote)

proc parseReply*(sim: Sim, seat: int, payload: JsonNode): Decision =
  case sim.turnKind(seat)
  of tkArgue: parseAdvocateReply(payload)
  of tkWhisper: parseJurorReply(payload)
  of tkVote: parseVoteReply(payload)

proc applyDecision*(sim: var Sim, seat: int, decision: Decision,
    scripted: bool) =
  ## Routes a decision to the right rule. Raises TribunalError on anything
  ## the rules reject.
  case sim.turnKind(seat)
  of tkArgue:
    sim.applyArgument(seat, decision.introduce, decision.argument,
      decision.notes, scripted)
  of tkWhisper:
    sim.applyWhisper(seat, decision.whisper, decision.lean, decision.notes,
      scripted)
  of tkVote:
    sim.applyVote(seat, decision.vote, decision.reason, decision.notes,
      scripted)

proc decisionJson*(sim: Sim, seat: int, decision: Decision): JsonNode =
  case sim.turnKind(seat)
  of tkArgue:
    %*{"introduce": decision.introduce, "argument": decision.argument,
      "notes": decision.notes}
  of tkWhisper:
    %*{"whisper": decision.whisper, "lean": decision.lean, "notes": decision.notes}
  of tkVote:
    %*{"vote": decision.vote, "reason": decision.reason, "notes": decision.notes}

proc actionFromEvent*(event: GameEvent): JsonNode =
  case event.kind
  of evArgue: %*{"introduce": event.cards, "argument": event.text, "notes": event.notes}
  of evWhisper: %*{"whisper": event.text, "lean": event.lean, "notes": event.notes}
  of evVote: %*{"vote": event.vote, "reason": event.text, "notes": event.notes}
  else: raise newException(Defect, "expected an installed action event")

# ---- The turn's batch -------------------------------------------------------

proc raisePolicyFailure(seat: int, cause, detail: string, rawReply = "",
    inconclusive = false) =
  var err = newException(PolicyDecisionError,
    "seat " & $seat & " policy decision failed (" & cause & "): " & detail)
  err.seat = seat
  err.cause = cause
  err.rawReply = rawReply
  err.inconclusive = inconclusive
  raise err

proc decideAll*(
  client: LlmClient,
  sim: Sim,
  seats: seq[int],
  prompts: seq[string],
  scripted: seq[ScriptKind],
  selection: PromptSelection,
  models: seq[string] = @[]
): seq[Decision] =
  ## One decision per seat in `seats`, in order — the whole turn as ONE
  ## parallel batch, because the seats decide simultaneously. Explicit scripted
  ## seats run their baseline; model seats require a configured LLM transport
  ## and raise PolicyDecisionError immediately on failure (no retry, never
  ## falling back to a scripted baseline). `prompts`, `scripted`, and `models`
  ## are indexed by SEAT.
  result = newSeq[Decision](seats.len)
  var open: seq[int]     ## indexes into `seats` still undecided
  for index, seat in seats:
    let kind = scripted[seat]
    if kind != skNone:
      result[index] = scriptedAction(sim, seat, kind)
    else:
      open.add(index)
  if open.len == 0:
    return
  if client.transport == ltNone:
    let failedSeat = seats[open[0]]
    raisePolicyFailure(failedSeat, "no_credentials",
      "no LLM endpoint or API key configured for model seat " & $failedSeat)
  var batch: RequestBatch
  for index in open:
    let seat = seats[index]
    let user = sim.userPrompt(seat, prompts[seat])
    let seatModel =
      if seat < models.len and models[seat].len > 0: models[seat]
      else: ""
    let request = client.requestFor(
      systemPrompt(sim, seat, selection), user, seat, seatModel)
    batch.post(request.url, request.headers, request.body, $index)
  let responses = client.curl.makeRequests(batch, client.timeoutSeconds)
  for position, index in open:
    let seat = seats[index]
    var rawText = ""
    try:
      rawText = client.textOf(responses[position].response,
        responses[position].error, batch[position].url)
      let decision = sim.parseReply(seat, parseJsonObject(rawText))
      var probe = sim
      probe.applyDecision(seat, decision, false)
      result[index] = decision
    except LlmTransportError as error:
      echo "tribunal llm: seat ", seat, " transport failure: ", error.msg
      raisePolicyFailure(seat, "transport_error", error.msg, inconclusive = true)
    except CatchableError as error:
      let cause =
        if rawText.len == 0: "request_error"
        else: "parse_error"
      let rawHead = snippet(rawText)
      echo "tribunal llm: seat ", seat, " failed (", cause, "): ", error.msg,
        (if rawHead.len > 0: " | raw=" & rawHead else: "")
      let detail =
        if rawHead.len > 0 and rawHead notin error.msg:
          error.msg & " | raw=" & rawHead
        else:
          error.msg
      raisePolicyFailure(seat, cause, detail, rawHead, inconclusive = false)
