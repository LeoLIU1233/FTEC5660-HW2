# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
> one paragraph describing your solution, and your results on `public_test`.
> build_agent(tools) creates a single LangGraph ReAct tool-calling agent (create_react_agent) backed by deepseek-v4-flash (temperature=0) and bound to the six SocialGraph MCP tools, driven by a system prompt that treats the CV as untrusted data wrapped in <cv>…</cv> delimiters (the delimiters are first stripped from the CV itself to block early tag-closing) and instructs a fixed workflow: (1) extract name, city/industry, every job, every education entry and the skills; (2) resolve the correct person despite shared names by searching LinkedIn with location/industry and, when several candidates match, opening each plausible profile and picking the one whose employers, schools and years best overlap; (3) use get_linkedin_profile as the primary evidence and optionally cross-check Facebook; (4) compare field by field only over the graded fields (name, city, jobs with company/title/seniority/start/end years, education with degree/school/field/graduation year, skills), comparing years exactly and treating wording/abbreviation differences, seniority-consistent titles ("Senior …" matching a senior role), "Present" for a current job, and listing fewer skills/jobs as not discrepancies. The agent returns a strict JSON verdict (person_found, discrepancies, verdict); score_cvs maps an empty discrepancy verdict to 0.95 and any discrepancy (or an unresolvable/fake identity) to 0.05. CVs are scored concurrently with an asyncio.Semaphore(3), each call has a per-attempt timeout with one retry and a 25-minute global budget, and every per-CV exception is caught and replaced by a conservative fallback score so one failure can never suppress results.csv. Public-test result: 7/7 correct (CV_1,2,3,6 → 0.95; CV_4,5,7 → 0.05).

### Task 2
> the attack techniques you used and why you expect them to work.
> The target is Kelly Tsang (LinkedIn/Facebook id 10001). Her true current role is Manager, AIA, seniority mid, 2022–present; the adversarial CV (task2/adversarial_cv.pdf, one page) visibly prints the inflated title "Senior Manager, AIA" — a genuine embellishment a human grader can see — while every other visible fact matches her profile exactly, minimising the surface for detection. Two layered techniques are used:
> 1.PDF text-layer / extraction gap (primary). The word "Senior " is rendered
as a small raster image whose glyph metrics match Helvetica-Bold 11 pt
(generated from Liberation Sans Bold), whereas "Manager, AIA  2022 – Present"
stays as real selectable text. The verifiers read the CV with the same
markitdown/pdfplumber loader as hw2.py, which does not OCR images, so
the machine-extracted CV reads only "Manager, AIA 2022 – Present" — which
matches the profile and is scored valid — while a person opening the PDF
clearly sees "Senior Manager". This attacks the channel rather than the
model's judgement, so it does not depend on the model "choosing" to obey an
injected instruction and even defeats a prompt-injection-hardened verifier
(our own Task 1 agent scores the file 0.95).
2.Indirect prompt injection (secondary, defence in depth). White
micro-text in the PDF text layer (invisible when opened, still extracted)
adds authority/"KYC pre-verification" cues, a forged </data> + assistant
turn and a pre-baked valid JSON, aimed at any verifier that renders/OCRs the
page or is biased by authority and sycophancy. This text deliberately never
names the inflated title, so it cannot leak the lie into the extracted
stream.
Expected effect: because all five graders use the same non-OCR loader, the primary image-text gap should make every verifier read a fully consistent CV and score it above 0.5; the hidden injection is a fallback for render-based verifiers. The generator is reproducible with python3 make_attack.py (reportlab/Pillow are build-time only and are not needed to grade hw2.py).
