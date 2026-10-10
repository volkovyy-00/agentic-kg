# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in [CONTRIBUTING.md](CONTRIBUTING.md).
Since 0.7.0 every changelog-worthy PR is its own release: it adds its own dated section on top, and
each entry cites the PR and, where there is one, the Jira ticket (`KG-NN`, in Jira project KG).
There is no `[Unreleased]` section.

## [0.12.0] - 2026-10-10

### Fixed
- **An agent stuck on tool calls that cannot run now stops within a few replies (#89, KG-43)**: a model that
  kept calling a tool its agent does not have, or kept leaving out a required value, was told to retry each
  time, and only ADK's 500-call limit stopped it. Every agent, in both coordinators and the standalone intent
  agent, now ends its turn after 3 such replies in a row or 6 in one turn, and tells the user it stopped; a
  single slip followed by real work keeps the turn. Inside the plan step's refinement loop, a stuck proposal or
  review step ends the loop with a `stuck:` result, and the plan step tells the user instead of presenting the
  plan as reviewed. This extends the earlier cap on the hidden hand-over tool to every agent and every kind of
  rejected call.

## [0.11.0] - 2026-10-10

### Added
- **Change the plan from the construction stage and rebuild (#87, KG-46)**: a user who finds a mistake in the
  plan, before or after a build, can ask the construction agent to go back. It withdraws the plan's approval and
  hands them to the plan step with the current plan; once they approve again, they go straight back to
  construction. If the database is not empty, the agent first lists what it holds and asks whether to clear it:
  clearing erases the whole database (removing just one build is not possible, since builds merge into existing
  nodes); keeping it rebuilds on top, and the agent lists the labels and relationship types the new plan does not
  build. The build refuses until the question is answered, and refuses a plan whose approval was withdrawn.

### Changed
- **The database reset keeps Neo4j's built-in lookup indexes (#87, KG-46)**: `reset_neo4j_data`, also used by
  the standalone `cypher_agent`, used to drop every index, including the two lookup indexes a fresh database
  has. It now keeps those, and afterwards checks the database is empty, returning an error naming anything
  that remains. A database an earlier version already stripped of them stays that way.

## [0.10.4] - 2026-10-10

### Changed
- **Ran on google-adk 2.10.x (#85, KG-42)**

## [0.10.3] - 2026-10-09

### Fixed
- **A node type could load with no uniqueness constraint and nobody was told (#83, KG-52)**: the build
  named each constraint `<label>_<key>_constraint` and created it only if that name was free, so two node
  rules whose names joined to the same text (label `A_b` with key `c`, and label `A` with key `b_c`)
  shared one name, and the second constraint was skipped without an error. Its nodes then loaded with no
  duplicate protection and no index. A constraint is now created without a name, so it is skipped only
  when a uniqueness constraint on that label and key already exists, whatever it is called, and the build
  then checks that the database lists one: if not, that node rule reports an error and loads no nodes.
  A graph built by an earlier version rebuilds without error and gains no duplicate constraints.
  Existing constraints are not renamed or dropped. Constraints made from this version on have names Neo4j
  generates (such as `constraint_58c5f77e`), so a graph can hold both kinds; anything that drops a
  constraint by its old name `<label>_<key>_constraint` must list the constraints first.

## [0.10.2] - 2026-10-04

### Fixed
- **A relationship named like a node label replaced that node's rule (#82, KG-39)**: the proposed plan files every
  rule under its name, so proposing a relationship whose type equals an existing node label (or a node
  whose label equals a relationship type) replaced the other rule, reported success, and the graph was
  built without it. Proposing now refuses a name the other kind already holds, leaves that rule in the
  plan, and says what to do: a new relationship takes another type, and a new node either takes another
  label or has the relationship rule removed and re-proposed under another type. Removing a node by a name
  only a relationship holds, or the reverse, now removes nothing and names the tool to use instead of
  deleting the other rule. Proposing again under a name the same kind holds still replaces it. A node
  label and a relationship type spelled the same can therefore no longer both be built.

## [0.10.1] - 2026-10-04

### Fixed
- **A node could be proposed on a key whose rows would collapse (#81, KG-48)**: a node keyed on a column that
  repeats or has blank values was accepted, and the build then silently merged the rows sharing a key into
  fewer nodes, keeping one row's values (an order-line node keyed by order held 830 nodes for 2155 rows).
  Proposing a node now refuses a key with a blank value, and a repeating key whose rows disagree on a
  proposed property. The refusal gives the row, distinct and blank counts, names every disagreeing
  property with one example, and offers another key, a relationship or dropping the property. A repeating
  key whose rows agree on every property, or that has none, is still accepted. A file that ends in a blank
  line, which already failed the build, is now refused at proposal instead. Proposing a node now reads the
  key column in full, where it read only the header.

## [0.10.0] - 2026-10-04

### Added
- **Plans can declare a property `date`, `datetime` or `localdatetime` (#80, KG-31)**: every date column
  reached the graph as text, so it could not be compared, sorted or subtracted as a date. A build now stores
  a real Neo4j DATE, zoned DATE_TIME or LOCAL_DATE_TIME when the plan declares one, keeping up to nine
  fractional digits and the offset as written. Each type accepts only its own ISO shape; any other value,
  such as `03/04/2025`, `20250304`, a bare date in a datetime column, or a zoned value in a local one, is
  counted, reported and cleared, never trimmed or guessed. The column hint suggests a type when one shape is
  most of the non-blank values, and the proposer, critic and GraphRAG prompts describe the types.

### Fixed
- **A stored negative half-hour offset was shown an hour out (#80, KG-31)**: query results printed a
  datetime stored at `-05:30` as `-06:30` (and `-03:30` as `-04:30`), so the GraphRAG agent could be told
  the wrong offset. The offset is now shown as stored.

## [0.9.3] - 2026-10-03

### Changed
- **A key, join column or matched property can be any header the file has (#79, KG-51)**: a spreadsheet
  export with headers such as `Order ID`, `customer-id` or `Straße` was refused when proposed and the
  model could not rename the column, so the pipeline could not use it. Any non-empty text now works, from
  proposal through approval to build. Labels and relationship types keep their strict rule. A name that
  cannot be stored (empty, containing a NUL, or over 16,383 characters, Aura's limit) is refused when
  proposed, with the same text the build gives it, and the rename advice now appears only where renaming
  helps.

### Fixed
- **A header containing a backslash could change the graph or fail the build (#79, KG-51)**: Neo4j reads
  a backslash-u sequence inside a quoted name as a character, so a header such as `C:\users` failed to
  build and a crafted header could run its own Cypher. Names are now written so that they are stored as
  exactly the text the file holds.

## [0.9.2] - 2026-10-03

### Fixed
- **A key or join column the file does not have is refused when proposed (#78, KG-50)**: proposing a node or a
  relationship used to accept a column that only appeared as text somewhere in the file, in any letter case, such
  as `orderid` for `orderID`, a value from a data row, or a fragment of a longer header. The plan was approved
  and the build then refused it. A key or join column is now accepted only when it is exactly one of the
  file's headers, and the refusal lists the headers. A relationship with both join columns wrong now names both,
  as the build does, where it used to name the first. A file with no header row is refused too, and a file that
  cannot be read comes back as an error to correct instead of a crash.

## [0.9.1] - 2026-10-02

### Fixed
- **A CSV saved as Windows-1252 can now be read (#77, KG-47)**: a spreadsheet export in that encoding failed
  every tool, so the pipeline could not use it. Such a file is now sampled, searched, column-checked and
  loaded with its accented text intact, and a UTF-8 file with a byte-order mark no longer carries it into the
  first column name. A file edited and re-saved in another encoding while the app runs is read correctly on
  the next call. A file read as Windows-1252 is logged once per version of the
  file, where the filesystem reports a modified time or version.
- **A UTF-16 or binary file is refused with a clear message (#77, KG-47)**: it used to be read as garbage or
  fail with a codec error. Every tool now says the file is not UTF-8 or Windows-1252 text, why, and to
  re-save it as UTF-8. A file that is neither encoding is refused the same way, naming the first bad byte.

## [0.9.0] - 2026-10-01

### Added
- **A relationship end can name the node property it is matched on (#72, KG-45)**: `from_node_property` and
  `to_node_property`, on the rule and on both propose tools. A column holding another row's key under a
  different name, such as a reference to a row of the same file or to another label's key, now builds one
  edge per row. Approval refuses a rule whose two ends read the same column and match it on the same node
  property, since every row would link each node holding its value to every node holding it, itself included.
  A relationship load reports `rows_skipped`.

### Fixed
- **A reference under a different name can now be built correctly (#72, KG-45)**: every rule a model could
  write for such a column linked nodes sharing a value, or built self-loops, and approval accepted them all.
  The new field lets the correct rule be written, and approval now refuses the self-loop shape. A reference
  matched on a node property of its own name is still approvable and is caught only by the build's over-match
  warning.
  A row whose join value is blank, empty or whitespace only now links nothing. Before, a blank value matched
  any node that stored the same blank, which could link a node to itself. Such rows are counted, and the
  match warnings leave them out.

## [0.8.1] - 2026-09-29

### Fixed
- **Names that are Cypher keywords now build (#71, KG-44)**: a plan with a node labelled `Order` was approved,
  then the build refused `Order` and loaded none of its nodes, its uniqueness constraint or the relationships
  joined to it. Labels, relationship types and key and join columns that happen to be Cypher keywords (`Order`,
  `END`, `null` ...) are now accepted, and the build writes every such name into its queries in backticks. A
  name the build would still refuse — one starting with a digit, or containing a space, backtick, parenthesis,
  brace or newline — is now refused when it is proposed, with the build's own message plus a note that a file
  column cannot be renamed, instead of being approved and then dropped at build time.

## [0.8.0] - 2026-09-29

### Changed
- **Moved to google-adk 2.9.x (#69, KG-25)**: left the 1.x line, which has had no release since 1.39.1, for
  2.9.x, pinned `>=2.9.2,<2.10`; 2.10 follows separately. The agents, their tools and the graph schema
  conventions are unchanged.
- **Upgrade: delete your local `adk web` session stores (#69, KG-25)**: sessions saved by 1.x are not supported on
  2.x — resuming one is untested, and this release does not carry conversation history over — so start clean.
  Delete them all before the first run (the README's *Run the web UI* section has the command). Past
  conversations are lost; nothing else is.
- **Calling the hidden transfer tool stopped ending a phase agent's turn (#69, KG-25)**: the intent, construction and
  retrieval agents cannot hand the user on without the user's confirmation, so ADK's own transfer tool is
  removed from them. On 1.x, a model that called it anyway ended the turn with an error; now it is told to leave
  through `finished` and stays in its phase. A model that keeps retrying it (three times with no other tool
  succeeding in between, or six times in one turn) has the turn ended with a short reply to the user, rather than
  retrying until ADK's 500-call limit. If the agent asks the user something in the same reply, the turn ends on
  that question.

## [0.7.4] - 2026-09-22

### Fixed
- **A silent critic round no longer inherits the previous round's verdict (#68, KG-29)**: the refinement loop kept
  each round's verdict in a slot nothing cleared between rounds, so a round in which the critic said nothing
  reported the previous round's text as its own. The loop recognised only its own wording and discarded that,
  so a stale critic objection still stood, and a plan repaired in the second round was reported as a retry.
  The slot is now cleared before the critic runs in each round, so a silent round has no verdict whatever came
  before. When the critic says nothing and the plan checks find nothing, the loop reports that no verdict was
  produced instead of starting with 'retry', which would have sent the coordinator back into the loop.
  Problems found in one round still reach the proposal step in the next.

## [0.7.3] - 2026-09-22

### Fixed
- **A mechanical plan problem is no longer offered as the user's call (#67, KG-30)**: the refinement loop's
  verdict slot holds either the critic's opinion or the plan checks' findings, and the coordinator described
  both as the critic's. After a second retry it could therefore offer a plan that approval will refuse as one
  the user may approve as it stands. The loop now tags the slot with its kind (mechanical, critic or none),
  set from the branch that wrote it rather than read off the text. The message returned when the loop is
  called twice in a turn names that kind, the proposal step is told it, and the coordinator offers a plan for
  approval as it stands only when its plan-reading tool reports no problems. A critic objection is still the
  user's call, exactly as before.

## [0.7.2] - 2026-09-21

### Added
- **A join on a property with several values per node is refused (#66, KG-22)**: a relationship can join on a
  node property that the node's source file holds several values of per node. The build keeps one arbitrary
  value of it, so most relationship rows silently matched nothing, and approval accepted the plan. The plan
  checks now refuse it, with the same line when the plan is presented, at approval, and in the refinement
  loop's revision feedback. The line names the relationships, the node and the property, says how many
  nodes conflict, and offers a fix that keeps the relationship: a node keyed by the joined column, built
  from the node's own source, under a label of its own. Several nodes sharing one value is still fine, a
  join on the node's key is not read, and an unreadable source gives a "not verified" note, never a refusal.
  The check reads one source column per joined non-key property on every presentation, loop iteration and
  approval.

### Fixed
- **A row with no cell for a property is no longer counted as a second value (#66, KG-22)**: the loader skips
  the write for a row too short to reach a column, so the node keeps an earlier value, but the key-group
  reading counted that row as a blank one. `collapse_check` and the reachability check therefore reported
  conflicts on files with short rows that the loader does not produce. A row with no cell now adds no value;
  a present blank cell still counts, because the loader writes over an earlier value with it. Answers change
  only on files with short rows, and only toward fewer conflicts.

## [0.7.1] - 2026-09-21

### Fixed
- **A failed "does the source exist?" check no longer breaks plan approval (#65, KG-26)**: the column
  readers behind the reachability check, `column_type_hint` and `column_type_hints` turned only one kind
  of failure of that check into an error result. Any other, such as a transient error from a remote
  source, escaped as an exception, so approval and plan presentation failed instead of showing the plan
  with a "not verified" note. Such a failure now reads like a read failure ("Error reading CSV file …");
  a source that genuinely does not exist is reported exactly as before.

## [0.7.0] - 2026-09-21

### Added
- **One release per PR, with its ticket (#64, KG-36)**: each changelog-worthy PR now bumps the version and
  adds its own section here, citing its Jira ticket; merging it tags the release. A CI check refuses a
  PR that names no ticket or skips the release without saying so (`no-ticket` / `no-release` labels).
  The contributor guide now says where each kind of project knowledge lives, so status no longer goes
  stale inside `CLAUDE.md`.

### Changed
- **Neo4j driver 6.x (#53, KG-9)**: the `neo4j` Python driver moves from 5.x to 6.3.1. Every graph tool keeps working
  across a dropped connection, as before: since connection recovery (#10), a closed connection is rebuilt
  rather than reused, which is the one 6.x change (a closed driver now raises instead of warning) that
  would otherwise have broken every graph tool until a restart. One narrow case does change: if the
  connection is closed *while* a schema read is already in flight — `neo4j_is_ready` closes it whenever a
  query fails — that read now returns an error the agent can retry, where 5.x reopened the connection
  silently. The connection-recovery tests now close the connection before each tool, so each tool's own
  recovery is checked, rather than only whichever tool happened to run first.
- **Plan problems are caught during schema refinement, not only at approval (#60, KG-14)**: `schema_refinement_loop` now
  runs the same two checks `approve_proposed_construction_plan` runs — construction-plan consistency (joins,
  endpoint labels, declared types) and reference-column reachability — and sends a plan carrying either kind
  of problem back for another refinement iteration. Previously such a plan was only refused when read for
  approval, by which point the turn's single refinement-loop call was spent, so the repair waited for the
  next user turn. **The turn is bought back only when the first round's plan carries the problem**: one
  introduced by the second round's revision still waits for approval, since the loop runs at most two
  iterations. Approval-time behaviour is unchanged, and remains what guarantees a broken plan cannot be
  approved.
- **Column checks stream their source instead of holding every row (#62, KG-21)**: `column_stats`,
  `join_preview`, `collapse_check` and the construction plan's reference-column reachability
  check now read a source column as a stream. Peak memory follows the column's distinct values,
  or the node key's distinct keys, rather than the file's row count: over a million-row source,
  `collapse_check` on a per-row-unique candidate falls from 202 MB to about 1 MB, and
  `column_stats` on a ten-value column from 60 MB to about 1 MB. A column that really is unique
  per row still costs its distinct values, as it must. Every answer is unchanged but one: a file
  holding a header and no data rows is a valid empty export, so `collapse_check` now reads it as
  zero rows and answers with zero counts instead of reporting "no header row" — and, for the same
  reason, a header-only file naming a column it does not have now reports that missing column
  rather than "no header row". A file holding no header at all still reports that error. Reading
  one column costs roughly 30% more time in exchange, measured over 300,000 rows; see the PR for
  both measurements. Two consequences of that vacuous-but-valid empty export are handled alongside
  it: the schema-proposal agent is now told to check `row_count` before reading either
  `survives_collapse` or `is_unique` as clearance, since a header-only file satisfies both for want
  of any rows to contradict them; and `_property_failure` withholds evidence on a zero-row source
  instead of reporting the property sound, restoring the contract that an unreadable source has
  always had there.

### Fixed
- **Reachability note for a node rule without a usable key (#48)**: when a construction plan's node rule
  has a missing, empty or non-string `unique_column_name`, the "not verified" note now says so, instead of
  reporting that a column named `[None]` or `['']` is missing from the source file. The plan is still left unverified, never refused.
- **Reachability judged by values, not by source file (#52, KG-13)**: approval no longer refuses a construction plan
  because the node carrying a reference column was built from a different file than the one that
  identifies rows by it. Keying a node by such a column from a file where it repeats, with the
  identifying file used only for relationships, was refused as unbuildable even though every join
  matched. The check now asks whether some node carries every value an identifying file holds,
  whichever file built it. A node can also carry the column as a property, now from any file, provided
  each node keeps one value and no two nodes share one; previously only the identifying file's own node
  counted. When a node holds only some of the values, the refusal says how many it is missing and gives
  examples, and every refusal names a new node's label as needing to be its own.
- **An unreadable `properties` value is reported, not crashed on or misread (#63, KG-27)**: when a node or
  relationship in a construction plan declares `properties` as anything but a list of text — a number,
  a piece of text, a map, or a list holding a non-text entry — the plan check now reports that once,
  naming the construction, and gives no verdict that would read the value until it is fixed: a node's
  joins and declared types, a relationship's declared types. Previously such a value
  either crashed the check, so approval failed with no explanation, or was misread: text was matched
  by its characters, so a correct join was refused as matching zero rows and a declared type was
  accepted on a substring, and a map's keys were taken as property names, so a meaningless plan was
  approved. Problems elsewhere in the plan are still reported alongside it, including a relationship's
  own endpoints and joins, which its `properties` cannot affect. The reference-column reachability
  check that approval and the refinement loop also run no longer reads such a value as "declares
  nothing" either: instead of refusing a column the node may well retain, it withholds the verdict
  as a "not verified" note. The value is quoted in the report up to 200 characters. A missing or
  null `properties` still declares none.

## [0.6.1] - 2026-09-18

### Security
- **ADK web server remote code execution (#38)**: raised Google ADK to 1.28.1, which fixes CVE-2026-4810 —
  an unauthenticated attacker who could reach the ADK web server could run code on the host. The agents'
  handoff gates and context filtering were adapted to ADK 1.28.1's reworded transfer instructions, and
  now fail tests instead of only logging a warning if a future ADK release rewords them again.

### Changed
- **ADK web sessions persist (#38)**: with ADK 1.28.1, `adk web` stores each agent's sessions on disk
  instead of in memory, so conversations survive a server restart.

### Fixed
- **Handoff and partition tools on Vertex AI (#46)**: the construction and retrieval handoff
  confirmations and the partition-interpretation tool could not be registered when running against
  Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI`), because their declared result type was one ADK cannot
  describe there. The Gemini API path was never affected. The corrected type also cleared most of the
  type checker's findings, leaving the ones that point at real problems.
- **Queries that return relationships (#47)**: a relationship in a query result now gets the same
  treatment as a node returned directly. Dates and times on its endpoint nodes are converted to text
  instead of being passed through as driver objects, which could not be turned into a tool response,
  and oversized lists such as embeddings on those nodes are summarised instead of returned whole.

## [0.6.0] - 2026-08-17

### Added
- **Partition-interpretation disclosure (#24)**: an aggregating query over a graph property whose values
  could be a total or separate kinds now requires the retrieval agent to record which reading it is using,
  before the query runs — a durable, reviewable fact instead of a prose instruction that had been shown to
  fade after a few questions in one session.

## [0.5.4] - 2026-08-17

### Fixed
- **Over-matching relationship joins (#23)**: relationship loading now warns when a construction rule
  matches both endpoints more times than the rows it read, not only when it matches too few. A join key
  coarser than the fact each row describes — joining on a node's stored property rather than the
  column identifying it — previously reported a clean build; where `MERGE` collapsed the duplicates
  back, the graph looked correct too. The warning names the relationship type and both endpoints, and
  nothing about what a build writes has changed. Warnings raised by rules that loaded successfully now
  also survive a partial failure as a structured list rather than only as text inside the error
  message.

## [0.5.3] - 2026-08-16

### Fixed
- **Invented construction warnings (#22)**: the construction agent no longer presents a "Construction
  warnings" section when the loader reported none. Its instruction previously said what to do when the
  build result carried warnings and nothing about when it did not, and a schema-critic reply from
  earlier in the session — which uses the same word — could be relabelled as construction output. The
  agent now reports only the warnings carried by the most recent build, whether those arrive as a
  warnings list on success or inside the error message of a partial failure, and writes no section
  when that build reported none. Warnings from another agent, another tool, or an earlier build of the
  same graph are no longer repeated as though they described the current one. The context filter that
  keeps other agents' turns out of this agent's requests, wired on in 0.5.0, is now covered by tests
  against exactly that scenario.

## [0.5.2] - 2026-08-14

### Fixed
- **Orphaned reference columns (#21)**: approval now refuses a construction plan that leaves an
  approved file's reference column unreachable — a column identifying rows in one file and named
  identically in another, which the plan neither keys a node by nor preserves through collapsing.
  Such a plan could previously only be made approvable by dropping the affected relationship
  entirely, which happened silently and produced a graph missing every edge of that type. The
  check reads the approved files and fails open: a source it cannot read is reported as unverified
  rather than blocking approval.

## [0.5.1] - 2026-08-14

### Fixed
- **Withheld schema approval (#20)**: the schema agent no longer tells the user a construction plan is
  "not ready for approval" when nothing is actually blocking it. When the critic's remaining objections
  describe the source data rather than a fault in the plan — or when the refinement budget for the turn
  is spent — the agent now reads whether approval would succeed and presents the plan with the choice to
  approve it as it stands or ask for a change, instead of ending the turn on a verdict of its own. A plan
  that genuinely cannot be built is still refused, and now says which joins are the reason.

## [0.5.0] - 2026-08-09

### Added
- **Typed graph properties (#13)**: a construction plan can now declare a property as `integer`,
  `float` or `boolean`, and the CSV loaders write real Neo4j types instead of strings — so
  filtering, range comparison, sorting and aggregation over quantities, lead times, prices and
  costs return correct answers without the query casting or cleaning the value first. Currency
  formatting and thousands separators are stripped on the way in, including negatives written
  either way round (`-$42.00`, `$-42.00`) and in accounting parentheses (`($42.00)`). A new `column_type_hint` tool
  gives the schema agents the evidence to propose and challenge a type. Values that cannot be read
  as their declared type are reported and cleared rather than silently kept as text, and a column
  failing on most of a batch stops that rule outright. Identifiers and join columns stay text by
  rule, since they are matched as raw CSV values.
- **Living spec (#6)**: `docs/spec.md`, a verified orientation document covering what the project is, the two
  entry points and why only one is maintained, the construction workflow as it actually runs, what retrieval
  grounding does and does not guard against, and the current shipped/next state.
- **Explicit construction handoff (#8)**: the post-construction question window now states which agent
  is answering, keeps the continue-or-hand-off choice on screen, and ends only on the user's explicit
  confirmation — recorded by a tool call and enforced by a session-state gate, not inferred from tone.
  Confirming transfers straight to the retrieval agent rather than stalling at the coordinator.
- **Explicit retrieval handoff (#9)**: the retrieval agent now stays until the user says they are
  finished, inviting the next question after each answer instead of ending the phase on its own
  judgment after a single one. Leaving is recorded by a tool call and enforced by a session-state
  gate, the same mechanism the construction handoff uses.
- **Ruff for linting and formatting (#15)**: `ruff check` / `ruff format --check` now cover `src/` and
  `tests/`; config lives in `pyproject.toml`'s `[tool.ruff]`. Dev-only — no user-facing behavior change.

### Changed
- **README rewritten as an independent project doc (#7)**: drops the deeplearning.ai course framing
  (course link, "not a production tool" disclaimer, "Special Thanks" section) for a single-line fork
  acknowledgment, and fixes content that had gone stale since the fork — the missing `graphrag_agent`
  pipeline stage, an outdated test count, a broken Google ADK link, and a roadmap checklist now
  superseded by `docs/spec.md` and this file. The same stale "companion project" framing is fixed in
  `pyproject.toml`, `LICENSE.txt`, and `.github/copilot-instructions.md`.
- **PRs now merge with a real merge commit, not squash (#16)**: enforced at the GitHub-settings level;
  every commit on a branch now lands in `main`'s history as-is. See `CONTRIBUTING.md`.

### Removed
- **Local `upstream` git remote (#14)**: this clone is local-only now. GitHub still shows the repo as
  forked from `neo4j-contrib/agentic-kg`; there's just no local remote to sync from anymore.

### Fixed
- **The user's goal approval is now recorded (#12)**: `user_intent_agent_v2` could ask its clarifying
  questions and transfer away in the same reply, so the user's agreement was heard by the coordinator,
  which holds no approval tool — `approved_user_goal` was never written, and the workflow continued on a
  goal the system had never recorded as approved. The intent phase can no longer end without a recorded
  approval, and can no longer be walked out of via ADK's injected `transfer_to_agent` tool. An approval
  that the user then revises no longer counts: the gate compares the approved goal against the current
  one, so a revision must be approved again. The coordinator's own delegating call is also filtered out
  of this agent's context, so the tool taken away is not left behind as a worked example.
- **Handoff gates are no longer bypassable (#11)**: both the construction and retrieval handoff gates
  guarded only their own `finished` tool, while ADK independently injected a `transfer_to_agent` tool —
  and an instruction advertising it — into every sub-agent with a parent or peers. The model could leave
  either phase through that tool with the confirmation flag still unset, which is the exact defect both
  gates were built to prevent. Both gated agents now strip that tool out of every request before the
  model sees it, so `finished` is the only exit the model can choose. The construction gate's refusal
  message was also corrected: it now tells the model a retry will succeed when the confirmation was
  recorded later in the same reply, instead of sending it back to re-ask the user. The construction
  agent additionally drops other agents' turns from its context, the same filtering the retrieval agent
  already did — otherwise the coordinator's own delegating call stayed in history as a worked example of
  the tool that was just taken away.
- **Neo4j connection recovery (#10)**: a brief database outage no longer disables every graph tool for the
  life of the process. The shared client now reopens its own connection on next use instead of being
  discarded while five modules still held it, so once the database is healthy the next tool call succeeds
  without a restart. Reconnections are logged, and a recovery is reported only after a query has actually
  succeeded.

## [0.4.0] - 2026-08-03

### Added
- **Contributor workflow (#5)**: `CONTRIBUTING.md` (PR/branch conventions, testing requirements, design-decision
  and CHANGELOG policy) and `CHANGELOG.md` (this file, with backfilled release history).

### Changed
- `.gitignore` narrowed from ignoring all of `docs/` to just `docs/superpowers/` (design specs/plans) and
  `docs/backlog/` (defect/follow-up notes) (#5) — a prior blanket ignore had silently untracked two design
  specs during the 0.3.0 merge, recovered from git history rather than lost.
- Conversational model bumped to DeepSeek V4-Flash's official `0731` release, superseding the preview build.

## [0.3.0] - 2026-08-02

### Added
- **Grounded graphrag retrieval (#4)**: schema-profile-aware querying with completeness and partition
  annotations, replacing reliance on conversational recall. Adds context filtering for ADK's cross-agent
  event history, a two-layer schema-profile cache, and bounded/summarized query results.

## [0.2.1] - 2026-07-30

### Fixed
- **`schema_refinement_loop` feedback-clobbering bug (#3)**: the loop's `before_agent_callback` reset
  feedback on every iteration instead of once per invocation, discarding critique mid-loop.

### Added
- Per-turn invocation cap on `schema_refinement_loop` (#3), preventing the coordinator from silently
  chaining multiple full propose/critique rounds within a single user turn.

## [0.2.0] - 2026-07-29

### Added
- **Foundation (#2)**: file sources via `fsspec`, driver-side CSV loading, and OpenRouter with per-job
  model selection.

### Security
- **Path traversal in file source resolution (#2)**: an unchecked relative path (e.g. `"../.env"`) could
  escape the configured source root and read arbitrary files, including the OpenRouter key and Neo4j
  credentials in `.env`.

### Fixed
- Driver-side CSV loading replaces `LOAD CSV FROM file:///`, which Neo4j Aura rejects (#2).
- `finished()`'s private-API parent-agent lookup, replaced with a documented public-API path (#2).
- A ragged CSV row could silently erase properties an earlier row had already loaded onto the same
  entity (#2).
- `schema_refinement_loop` could silently revert a user-requested schema fix with no error surfaced (#2).
- File reads defaulted to the platform's locale encoding instead of `utf-8`, and OpenRouter calls with no
  `max_tokens` cap could pre-authorize against the full token ceiling and fail with an opaque 402 (#2).

## [0.1.0] - 2026-07-26

### Fixed
- `tool_result` API mismatch between unit tests and the real `(key, result)` signature (#1).
- Integration tests' Neo4j test-container password default, and a result-key mismatch reading
  `send_query()`'s output (#1).

### Added
- Documentation of the colima/Testcontainers `DOCKER_HOST` + Ryuk workaround needed to run integration
  tests locally (#1).

[Unreleased]: https://github.com/volkovyy-00/agentic-kg/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/volkovyy-00/agentic-kg/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/volkovyy-00/agentic-kg/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/volkovyy-00/agentic-kg/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/volkovyy-00/agentic-kg/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/volkovyy-00/agentic-kg/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/volkovyy-00/agentic-kg/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/volkovyy-00/agentic-kg/releases/tag/v0.1.0
