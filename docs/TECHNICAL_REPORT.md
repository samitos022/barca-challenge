# All Blacks FC technical report

More than a Hack 2027, ball possession in Barcelona vs Athletic Club. Submission file:
`evaluation/results/v2cml3_possession.csv` (submitted as `All Blacks FC.csv`, this document as `All Blacks FC.md`).

**In one paragraph.** We label every frame from the provided tracking (ATD) only. A hand-written
rule machine (v2) reads touches, control and stoppages from the tracking. A gradient-boosted
classifier then takes v2's answer plus about 150 other measurements and gives, for every frame, the
probability of home possession, away possession and dead ball. A Viterbi decoder turns these
probabilities into possession spells. Defects found by watching BAR-ATH (the ball vanishing in a
keeper's hands, corners and penalties with the taker out of the image) were turned into general
signals and training conditions, and kept only if they held out of fold. The classifier never sees a labelled frame of the
competition match. It is trained on 89 full matches from four open datasets: DFL Bundesliga,
Metrica, SkillCorner and the PFF World Cup 2022. Their tracking is degraded to look like
BAR-ATH's, using statistics of BAR-ATH's tracking that need no labels. Out of fold on those 89
matches the final model scores spellF1 0.704 (macroF1 0.915). On the same matches v2 alone
scores 0.520 and the official baseline 0.327.

---

## 1. Method and possession rules

### 1.1 Pipeline

The submission is produced by eight steps in sequence, each working on the output of the previous one.

#### 1.1.1 Tracking in metres (`tracking.py`)

- **What the ATD gives**: for every frame, each tracked slot (46 in the tactical ATD) and the
  ball, as fractions (0-1) of the pitch.
- **Conversion**: scaled to the 105.3 x 68.0 m pitch of `games.py`. Row i is video frame i
  (`frame_id - 1`), so every later step works on the video's own timeline.
- **Missing values**: a position the tracker does not have is NaN; nothing is interpolated.
- **Ball speed**: computed from consecutive positions.
- **Attacking direction**: taken in each half from the mean x of each team's players.
- **Cache**: the result goes to `data/derived/`.

#### 1.1.2 Team reassignment (`team_reassign.py`)

**The problem.** When the tracker loses a player and finds him again, the new stretch of track
can carry the other team's label. The typical case is a goalkeeper read as an opposing attacker. On
BAR-ATH a track ends and, within half a second and at the same spot, a track of the other team
starts 78 times.

1. **Pieces and chains.** Every slot is cut into pieces: tracked without gaps and without jumps
   over 1 m in one frame. A piece that starts within 12 frames of where another ended (0.5 m, plus
   0.1 m per frame of gap) continues it. A chain of pieces is one physical player.
2. **Context model.** A boosted classifier looks at a player on a frame without his label and says
   which team he plays for. It uses:
   - depth from both goals and rank inside each team's block;
   - distance from the last man and from the defensive line;
   - nearest neighbours, and pace compared with each team;
   - how many players each team already has, and the ball;
   - the same measurements averaged over the 5 s before and after.

   It is trained only on 10 external matches (DFL, Metrica) degraded like BAR-ATH and is symmetric
   in the two teams. Alone it guesses 92.9% of positions.
3. **Decision.** For each chain, a dynamic programme picks the cheapest sequence of teams. It
   weighs the context evidence against a cost of 6 for contradicting the tracker on a piece and a
   cost of 6 for changing team along the chain. Evidence stops growing after 10 s of a piece: what
   counts beyond that is the average, not the length, because a role can look wrong for a long
   time.

**Results.** Out of fold on the external matches, positions in the wrong team fall from 0.85% to
0.04%, and from 1.99% to 0.04% with goalkeepers swapped on purpose. With reassigned teams, v2 on
those matches scores exactly what it scores with the true teams (DFL 0.340 → 0.346, Metrica
0.572 → 0.587). On BAR-ATH 40 pieces change team: 655 s of player tracking, 0.5% of positions.

#### 1.1.3 Frozen positions removed (`track_fixes.drop_frozen`)

**The problem.** When a player leaves the camera image, the tracker keeps him at his last
position:
- 107 such stretches, median 32 s, the longest 348 s;
- 4.6% of all positions;
- 0.05 m/s against 1.41 m/s for players in view.

The main dead-ball cue is the players' pace breaking, so these still "players" pull the pace down
and make live play look stopped.

**The fix.** Every position is projected into the image with the provided tactical homography. A
position outside the image that has moved less than 0.5 m/s for more than 2 s is removed, and so
are positions in view that are perfectly still (under 1 mm per frame). The mean pace of the
remaining players rises by about 5%.

Steps 1.1.2 and 1.1.3 together give the source called **`tactical_clean`**.

#### 1.1.4 Whistle (`whistle.py`)

**The detector.** For every frame of the tactical video's audio it:
- takes the spectrum between 2 and 4.5 kHz;
- subtracts from every frequency its own median over 10 s, which removes steady tones;
- keeps narrow tones at least 15 dB over that background, stable in pitch, lasting at least 0.2 s.

**How v2 uses it.** An event adds 0.3 × the players' braking in the following 2 s to the dead
score, so a tone counts only if the players actually stop.

**Measured limit.** The tactical microphone mostly hears the crowd: about 11 events a minute.
The cue changes 0.1% of the final frames (section 2).

#### 1.1.5 Rule machine (`possession_v2.py`)

The hand-written state machine described in section 1.2. It runs on the `tactical_clean` arrays
and returns two things:
- an H / A / D label for every frame;
- its internal state: the challenger's possession score, the dead-ball score, the restart score
  and the points of each rule.

Both are inputs of the next step. v2's label is not the submission: it is one of the measurements
the model learns to trust or not.

#### 1.1.6 Measurements and three-state classifier (`possession_ml.features3`, `prob3`)

For every frame, 151 measurements are computed from positions and from v2's output (section 1.4).
The trained classifier (`data/derived/possession_ml3.pkl`, learnt only on the 89 external matches)
turns them into P(home), P(away), P(dead).

Each frame is read twice, the second time with the two teams exchanged, and the two answers are
averaged. The probabilities therefore cannot depend on which side is Barcelona.

#### 1.1.7 Viterbi decoder (`possession_ml.decode3`)

**Why a decoder.** Frame by frame the probabilities flicker (60% home, then 55% away). Taken one
frame at a time they would give hundreds of one-frame possessions.

**What it does.** It finds, separately in each half, the sequence of states with the highest total
log-probability once a cost is paid for every change:
- **16 per change of possession.** With the new team twice as likely as the old one on every frame
  (ln 2 ≈ 0.69), a change needs about 23 frames (0.9 s) of evidence. Ten times as likely: about
  7 frames (0.3 s).
- **32 per change between live and dead ball**, twice the evidence. A false stoppage splits a
  possession in two, and spellF1 punishes that.

The costs were chosen out of fold on the external matches; anything from 4 to 24 gives nearly the
same score.

The halves are the graded blocks of `games.py`: frames 11,185-90,651 and 110,808-190,197.

#### 1.1.8 Output file

`baseline.write_submission` writes the header `frame,label` and one row per video frame
(199,319). Every frame outside the two halves is `X`. The file is checked with
`evaluation/validate_submission.py` before submission.

### 1.2 The rule machine (v2)

v2 is written by hand and is fully inspectable. Every frame of its diagnostics breaks its score
down into per-rule points that add up exactly to the score.

- **Change of possession.** The team without the ball accumulates points and the team with the
  ball removes them. Points decay with a half-life of 1 s. At 1.0 possession changes, back-dated to
  the start of the episode of evidence (not to the moment of the decision).
  - **Control**: the nearest player to the ball, within 2-3.5 m, with a 1 m margin over the
    opponent, and the ball slower than 4 m/s.
  - **Touch**: the ball's velocity changes by at least 3.5 m/s and the nearest player of that team
    is within 3 m.
  - **Confirmation**: who controls the ball in the next 2 s.
  - **Two touches in a row**: two touches by the team without the ball, 0.4-3 s apart, with
    nothing from the other team in between, are a change from the first touch ("a team completes
    passes").
  - **A fast ball** (above 8 m/s, no deflection) is in flight. It does not count as control.
- **Entry into dead ball.**
  - **Ball out**: beyond a line by more than 1 m for 3 frames.
  - **Dead score** (threshold 1.0), built from three cues:
    - the 4 nearest players far from the ball (one-sided: players gathering around the ball after
      a foul do not cancel it);
    - **the pace breaking**: the players' mean speed over the next second against the previous
      4 s. This is the ratio, not the absolute speed, because slow build-up play is slow but does
      not brake;
    - the whistle, at a small weight.
  - **A cause is expected**: a challenge, an offside on the last pass, or a ball reappearing out.
    Without one the threshold rises to 1.25.
  - **Long ball gap** (rule 8): the ball is untracked for at least 5 s while the players walk. It is
    dead only with a second cue: lost near a line, reappearing on a line or a set-piece spot, pace
    breaking, players far, or whistle.
  - **Set-piece layouts read from the players alone** (rule 13): a wall of 3 still team-mates,
    a corner (a slow player at the flag and at least 6 in the box), a kick-off (everybody in their
    own half). On the external matches these frames are dead 92-99.7% of the time.
  - **Goalkeeper holds** (rule 12): a ball gap that starts in or near a penalty area, lasts 2-20 s,
    does not end at a restart spot, with the players not walking, is possession of the defending
    team, not a stoppage.
- **Restart.** The first touch with the ball in play after at least 1 s, dated to the ball's first
  movement. Without a visible touch, the players running again with the ball in play. The first
  restart of each half needs the ball within 4 m of the centre spot.

### 1.3 Where we drew the lines

- **Controlled possession** needs control or repeated touches by the same team. A single touch
  that the other team gets back within about 2 s (rebound, block, challenge) does not change
  possession. This follows CHALLENGE.md ("a simple touch without control").
- **Loose balls and passes in flight** keep the last team in possession until the other team
  controls or touches the ball twice.
- **Challenges** are neither a change of possession nor a dead ball by themselves. They are a
  *cause* that makes a following stoppage more likely.
- **Restarts** belong to the team of the player who plays the ball. The dead ball ends at the
  ball's first movement, not at the referee's whistle. Ball carried to the spot by a player during a
  stoppage is still dead ball. The learnt model was trained on simulations of exactly this case
  (section 3.2).
- **Missing tracking.** A missing ball is never a stoppage by itself. The decision comes from the
  players: pace, layout, where the ball vanished and where it comes back. Halftime and the frames
  outside the two halves are `X`.

### 1.4 The learnt three-state model

- **Inputs (151 per frame, all computed from positions):**
  - Possession geometry for each team: distances to the ball, control and touch counts in windows
    of 0.5-2 s, width and depth, ball position and velocity in the attacking direction.
  - v2's label and its internal scores.
  - Pace and pace breaking over 2-10 s, players far from the ball, distance of the ball from the
    lines, the set-piece layouts.
  - The ball gap the frame is in: its length, the pace inside it, and where the ball vanished and
    reappears.
  - Whether the ball near a player is contested.
  - Time since the last ball out, layout or v2 stoppage.
  - 23 **goalkeeper signals** (`keeper_signals.py`, stage 6 of section 4). For the gap the frame is in, seen
    from each keeper:
    - distance of the vanished ball from the keeper, and whether it vanished in his box;
    - how far he has moved from his goal line since, and his speed;
    - opponents leaving the box;
    - speed, direction and keeper distance of the ball when it reappears (a throw or a kick).
  - 24 **set-piece signals** (`setpiece_signals.py`, stage 7), read from the players alone and
    saying *which team* will restart. After a stoppage "there is a clear possessor" (CHALLENGE.md):
    the team that takes the restart owns the ball from its first touch, even if the other team
    clears it a second later. On BAR-ATH the corner taker is often outside the image (the far
    flags are in view on 2% of frames) and the ball is often missing, so the layout of the other
    players is the only evidence. For each team:
    - **corner**: a slow player of the team at a flag of the end it attacks, with that box full;
    - **crowded box**: at least 9 players in the box the team attacks, at least 3 of each team,
      median pace under 1.3 m/s for 2 s (a corner or a free kick waiting, even with the taker
      untracked);
    - **penalty**: one player of the team within 9.15 m of the spot (the kicker, who stands back
      for his run-up), at most one player on the goal line between the posts (the keeper), nobody
      else in the box or the arc (1 m of tolerance at their lines, for tracking noise), at least 7
      players gathered just outside it, everybody slow for 1.5 s. Or the same with the kicker
      untracked: box and arc empty but for the other team's keeper on his line, and at least 3
      players of each team at the edge of the box;
    - **throw-in**: the slow player of the team furthest out on a touchline;
    - **wall**: a wall of 3 standing players of the other team (a free kick for this one);
    - for each: whether it is on, and the seconds since it was last seen; for all together, the
      share of the last 5 s with any, and the seconds until the next one.
- **Symmetry.** Every input exists for "home" and "away". Training rows are duplicated with the
  teams exchanged, and at prediction the two readings are averaged. The model cannot learn that one
  side is Barcelona.
- **Classifier.** scikit-learn `HistGradientBoostingClassifier`: 200 iterations (600 trees, three
  classes), 63 leaves, at least 200 rows per leaf, L2 1.0, learning rate 0.1, `random_state=0`.
- **Decoder.** A three-state Viterbi with log-likelihood costs of 16 per change of possession and
  32 per change between alive and dead, chosen out of fold (section 5.1). The cost sweep from 4 to
  24 is flat.

Why this design rather than an end-to-end network: the labelled data that exists for other
matches is a few dozen full games. The signal that matters (who touched the ball, did the play
stop) is well described by a few dozen physical measurements. A boosted tree over those, plus a
decoder that knows possessions last seconds, trains in about 1.5 h on a laptop, can be checked
match by match and read. The rules remain the main input and the model learns how much to trust them on each
kind of tracking.

---

## 2. Assets

| asset | use |
|---|---|
| Tactical ATD tracking | **The only input of the labels.** Converted to metres; row i = video frame i. |
| Tactical homography | Only to know which positions are inside the camera image: removing the frozen players out of view, and the bank of camera framings that the `view` training condition applies to the external matches (section 3.2). |
| Tactical video, audio track | Whistle-like tones (`whistle.py`), a weak cue in v2 (see below). |
| Tactical video, images | Not used by the pipeline. Watched only to understand errors (no labels, no frame-specific constants). |
| Broadcast video, broadcast ATD and homography | Tried and dropped from the final file (below). |
| Smart Tagging | Tried and dropped (below). |
| Public 5 minutes of ground truth | Only to check the pipeline end to end (format, alignment, label meaning). Never used to choose a threshold or a version. |

**Alignment and coordinates.** ATD coordinates (0-1) are scaled to a 105.3 x 68.0 m pitch. The
attacking direction of each team is taken in each half from the mean x of its players. The two
graded blocks are the halves as given in `games.py`.

**Corrections of the provided tracking, all by code:** team reassignment (section 1.1.2) and frozen
positions removed (section 1.1.3).

**Tried and dropped:**
- **Broadcast fusion** (`sync_broadcast.py`, `fusion.py`). Synchronisation is fully automatic:
  - Apple Vision OCR of both on-screen clocks gives candidate offsets;
  - audio correlation refines them;
  - player-to-player matching confirms each frame as "live".
  Result: +4014 / +26920 frames, live 58% of the broadcast, median player distance 0.26 m.
  The fused ball raises the share of frames with a ball from 59.3% to 63.0%, but the fused
  players change nothing. With BAR-ATH's two versions equal on the public check and no external
  match with a broadcast to validate on, we kept the simpler, cheaper pipeline. The fused version
  needs about 30 minutes of OCR and has more steps that could break exact reproduction.
- **Re-detecting the ball** in the gaps:
  - Roboflow `sports` YOLOv8 on a 640 px crop recovers 59.3% → 63.2% coverage;
  - WASB-SBDT does not transfer to this camera.
  With no external video with known truth, we could not tell recovered balls from false
  positives. Not adopted.
- **Referee whistle.** The tactical microphone mostly hears the crowd: about 11 tone events per
  minute. After an event the players brake no more often than at random (12-18% against 18%). A
  pretrained sound classifier (PANNs CNN14) gives the "whistle" class zero probability on both audio
  tracks. The cue stays in v2 at a low weight and changes 0.1% of the final frames.
- **Smart Tagging** (Metrica automatic tags) as a hint to the decoder. Its phases agree with the
  model 90-97% of the time. Used as a hint, it merged quick changes of possession (111 → 103
  possessions in the first half on review), so it was removed.
- **Replay detection** in the broadcast: diagnostics only.

---

## 3. Training data and models

### 3.1 Datasets (none from the provided game)

| dataset | matches | tracking | labels H / A / D, and who produced them | source and licence |
|---|---|---|---|---|
| DFL / IDSSE Bundesliga 2022/23 | 7 | TRACAB optical, 25 fps | dead ball and team from TRACAB's `BallStatus` and `BallPossession` (provider data shipped with the open dataset) | Bassek et al., figshare [10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0 |
| Metrica sample data | 3 | Metrica, 25 fps (same provider as the competition) | from Metrica's sample event data (events annotated by Metrica): team from the possession events, dead ball from the stoppage events | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) |
| SkillCorner open data, A-League 2024/25 | 20 | broadcast tracking, 10 fps, interpolated to 25 | alive = inside one of SkillCorner's "phases of play"; team from the phase; stoppage start refined to the ball crossing the line | [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT |
| PFF FC / Gradient Sports, World Cup 2022 | 59 of 64 | broadcast-derived, 29.97 → 25 fps | from PFF's event data (OUT / END = dead, kick-off and on-the-ball events = alive, `home_ball` = team), annotated by PFF analysts on the video | public Google Drive folder linked from the [kloppy documentation](https://kloppy.pysport.org/user-guide/loading-data/); no licence file shipped |

- **We did not label anything by hand**, on any game. All labels above were produced by the data
  providers. We only convert them to H / A / D with a fixed rule per dataset (in
  `solution/external/*.py`).
- **Amount: 89 full matches.** The final training set is about 4.8 M frame rows (336 runs: every
  match in each condition of section 3.2), doubled by the team swap: one frame in 5 on DFL, Metrica and SkillCorner, one in 15 on PFF, and only frames with at
  least 10 tracked players.
- **Dropped datasets**: SoccerNet-GAR and FOOTPASS (NDA), SoccerTrack v2 (ball rebuilt from
  events), SkillCorner 2020 (no alive / dead), Signality (credentials not public), Kaggle World Cup
  2026 (ball every 5 s).

### 3.2 Making the external tracking look like BAR-ATH's

BAR-ATH's tracking is much worse than TRACAB's: the ball is missing 41% of the time, it is
blurred and lags, and players are missing or frozen. A model trained on clean tracking would
learn cues that do not exist here, so each external match is degraded (`solution/external/degrade.py`).
Every target statistic is measured on BAR-ATH's tracking **without labels**:

- **players**: removed and frozen in the same proportions;
- **ball gaps** (`degrade_ball`): 41% missing, median gap 10.9 s, players' pace inside gaps
  1.27 m/s, where the ball was last seen (line / box / elsewhere 38 / 20 / 42%, BAR-ATH 35 / 21 / 44);
- **ball shape** (`shape_ball`): 6-frame blur, 0.7 m slow position error, 4 cm jitter, fitted to
  BAR-ATH's touch rate (14.2 / min), touch distance (1.62 m against 1.46) and velocity jumps.

Four conditions of every match enter the training:
- `realistic`: the degradation above;
- `moved`: the ball is carried by the taker to the restart spot during stoppages, as it is visible
  in BAR-ATH;
- `keeper`: the ball is removed while a goalkeeper holds it (`keeper_holds`, `hide_keeper_holds`).
  This is the defect we saw in review: the ball vanishes in the keeper's hands and the model
  called it dead ball. The condition covers 12-27 holds per match.
- `view`: only what BAR-ATH's tactical camera would show (`camera_view`). From BAR-ATH's own
  tactical tracking and homography (no labels) we keep a bank of camera framings, each indexed by
  the centroid of the players it shows. Every frame of an external match takes one of the 20
  framings nearest to its own players' centroid (which one is drawn every 5 s, with a fixed seed),
  and the players and ball outside that image are removed. A DFL match then keeps 19.8 tracked players per
  frame, against BAR-ATH's 19.5. This is the case of the review frames where a corner or a penalty
  is clear from the layout while the taker and the ball are out of view.

SkillCorner and PFF keep their own broadcast tracking, which is already imperfect, instead of the
degradation; the other conditions are applied on top of it (SkillCorner has no `moved` run).

### 3.3 Models

| model | learnt from | file |
|---|---|---|
| team reassignment | 10 external matches (DFL, Metrica), degraded | `data/derived/team_reassign.pkl` |
| three-state classifier (final) | 89 external matches in the conditions above | `data/derived/possession_ml3.pkl` |

No pretrained network is used by the submission pipeline.

---

## 4. How we worked

**Rules we set ourselves** (some stricter than RULES.md):
- Nothing is chosen on the public 5 minutes. They hold 17 possessions, so spellF1 there moves by
  about ±0.33 for one possession more or less.
- Every change is measured on the external matches, out of fold, with matches grouped:
  - 5 folds by match, round-robin within each dataset;
  - all conditions of a match in the same fold;
  - parameters chosen on half of the matches and scored on the other half, then swapped.
- The public window is only scored after a decision, to check that nothing broke.
- We watched review videos of BAR-ATH beyond the public window to find defects. Every defect
  became a general mechanism, tested on the external matches. Nothing was tied to a frame of the
  game.

**Stages, with the out-of-fold spellF1 where it was measured:**

1. **v1**: possession changes only, scored from control and touches.
2. **v2**: dead ball added (ball out, pace breaking, gaps, set-piece layouts), with restarts dated
   to the ball's first movement.
   - Calibrated first on the 3 Metrica matches (0.550 → 0.663 with their own tracking), then on
     the realistic DFL bench: DFL 0.30 → 0.34, Metrica 0.58.
   - An oracle analysis showed that the points are lost on *who has the ball*: a perfect team
     brings DFL to 0.76, while a perfect dead ball only brings it to 0.43.
3. **Learnt team**: a classifier for the team inside v2's live spells. DFL 0.33 → 0.45,
   Metrica 0.57 → 0.66.
4. **Three-state model**: the classifier also decides the dead ball. DFL 0.558, Metrica 0.689.
   - **Second feature set**: is the ball contested, plus longer windows. Trained on the `moved`
     condition after review showed fake possessions while the ball was carried to the spot.
     Fake possessions per stoppage 0.28 → 0.07.
   - **SkillCorner added**: on SkillCorner, with those matches never seen 0.692, in training 0.735.
   - **PFF added**: 10 held-out World Cup matches 0.752 → 0.832.
5. **Research on spellF1**, measured on all 89 matches (mean of the 4 datasets, base 0.703):

   | change | result | adopted? |
   |---|---|---|
   | asymmetric decoder costs | 0.702 | no |
   | second stage on the probabilities (MS-TCN-like) | 0.707, but DFL −0.017 | no |
   | change-point detectors (ASRF-like) | 0.703 | no |
   | logistic correction from collective signals | 0.704 | no |
   | 180 collective team signals in the model | 0.721 | no |
   | more frames per match (1 in 20 to 1 in 2) | 0.702-0.706 | no |

   The collective signals looked best but failed the cross-provider test (train on three
   datasets, test on the fourth): SkillCorner 0.710 → 0.663. They learn the habits of each
   tracking system.
6. **Goalkeeper holds**, out of fold:

   | | spellF1 | holds called dead: SkillCorner / PFF / Metrica / DFL |
   |---|---|---|
   | before | 0.703 | 11.9 / 11.8 / 34.7 / 6.0% |
   | holds simulated everywhere | 0.701 | 5.6 / 6.1 / 31.1 / 5.2% |
   | holds + keeper signals | 0.704 | 4.2 / 4.1 / 32.3 / 5.2% |

   - This version scores 0.704 with the holds hidden as well.
   - On BAR-ATH, without labels: 8 ball gaps start within 5 m of a keeper in his box (65 s). The
     share of those frames labelled dead goes from 52% to 33%. The dead-ball share of the match
     and the number of possessions are unchanged.
7. **Set pieces and the camera's framing** (final change). Review showed two more defects: after a
   corner or a throw-in the first, short possession of the restarting team was missed, and a corner
   or a penalty that was obvious from the players' layout was not read because the ball (and often
   the taker) was out of the image.
   - **Set-piece signals** (section 1.4) say which team will restart. Checked on the external
     matches, realistic tracking (`analisi/calci_piazzati.py check`):

     | layout | on a dead ball | its team takes the next restart |
     |---|---|---|
     | corner | 88-91% | 92-98% |
     | crowded box | 61-85% | 66-83% |
     | throw-in | 64-79% | 78-86% |
     | wall | 95-100% | 67-76% |
     | penalty | 75-100% | 58-87% |

   - **The `view` condition** (section 3.2) puts the external matches in BAR-ATH's situation, where
     these signals matter: on the external tracking the ball is almost always there, so the model
     already knows who takes the corner.
   - **The penalty rule took three versions.** One player on the spot fired on goal kicks. One
     player inside the arc (the kicker stands back for his run-up) found the 3 true `Penalty`
     events of the DFL matches but never fired on BAR-ATH: in one case two players stood exactly
     on the arc's line and the tracking noise put them inside, in the other the kicker was not
     tracked. The final rule adds 1 m of tolerance on the arc and the variant without a kicker
     (section 1.4). On DFL it still finds the 3 penalties and raises 7 other alarms, all on dead
     balls (5 free kicks, a goal kick, a throw-in) and none before a kick-off; on BAR-ATH it fires
     on the two cases, once per team, and nowhere else.
   - Out of fold, mean spellF1 of the four datasets:

     | | realistic | keeper | view | first team after a restart | short restart possessions found |
     |---|---|---|---|---|---|
     | keeper model (stage 6) | 0.704 | 0.704 | (not tested) | 91.6-98.0% | 63.5-76.5% |
     | + set-piece signals | 0.702 | 0.703 | 0.693 | 91.6-98.2% | 64.5-76.4% |
     | + `view` in training | 0.705 | 0.705 | 0.699 | 91.5-98.0% | 63.2-76.6% |
     | + final penalty rule (**final**) | **0.704** | **0.705** | **0.700** | 91.6-98.0% | 62.7-76.5% |

     The last two columns are realistic tracking, ranges over the four datasets. The set-piece
     signals alone change nothing where the ball is visible. With the `view` condition in training
     they gain 0.006-0.007 on camera-framed tracking (Metrica 0.681 → 0.696, SkillCorner
     0.725 → 0.736, PFF 0.839 → 0.848, DFL 0.529 → 0.520) and lose nothing on the other two. The
     last two rows are even (differences of 0.001); we kept the final penalty rule because it is
     the one that reads BAR-ATH's two penalty layouts.
   - On BAR-ATH, without labels, against the file of the keeper model: the team of a corner layout
     has the first possession after the stoppage in 12 of 14 cases (before 10 of 13), crowded box
     22 of 28 (20 of 29), throw-in 41 of 48 (39 of 46). The probable keeper holds labelled dead go
     from 32.5% to 43.9%, a cost we did not manage to remove. The dead-ball share of the match goes
     from 45.5% to 46.1%.

---

## 5. Results

### 5.1 External matches (out of fold, `analisi/classifica_generale.py`)

spellF1 / macroF1, realistic tracking:

| version | DFL (7) | Metrica (3) | SkillCorner (20) | PFF (59) | mean |
|---|---|---|---|---|---|
| official baseline (smooth50) | 0.230 / 0.588 | 0.319 / 0.786 | 0.360 / 0.661 | 0.398 / 0.740 | 0.327 / 0.694 |
| v2 rules | 0.341 / 0.744 | 0.570 / 0.853 | 0.548 / 0.734 | 0.621 / 0.846 | 0.520 / 0.794 |
| three states, before the keeper work | 0.542 / 0.884 | 0.689 / 0.900 | 0.734 / 0.920 | 0.848 / 0.947 | 0.703 / 0.913 |
| + keeper holds and signals | 0.533 / 0.881 | 0.692 / 0.904 | 0.737 / 0.921 | 0.853 / 0.950 | 0.704 / 0.914 |
| **final** (+ set pieces, `view` in training, final penalty rule) | 0.532 / 0.883 | 0.694 / 0.905 | 0.739 / 0.921 | 0.852 / 0.951 | **0.704 / 0.915** |

On camera-framed tracking (`view`, the closest to BAR-ATH) the final model scores 0.700, against
0.693 without the `view` condition in training. On realistic tracking the set-piece work is even
(0.704 before and after): its gain is where the camera hides the taker.

DFL is the hardest set: its labels have many very short possessions (30% under 2 s), which a
decoder built to avoid flicker merges.

### 5.2 Public 5 minutes (pipeline check only)

The final file scores spellF1 0.652 and macroF1 0.898 on the public window, with 10 possessions
against 17 true. The official baseline `baseline_smooth50` scores 0.326 / 0.621 there. Other versions
we built score 0.627-0.658. With 17 possessions those differences are one or two possessions, so
they were not used to choose.

On the whole match the final file gives:

| | first half | second half |
|---|---|---|
| H / A / D | 34.9% / 20.4% / 44.7% | 30.7% / 21.9% / 47.5% |
| possessions | 110 | 127 |

That is about 4.1 possessions per minute of effective play, close to the 4.2 of the public window.
The dead-ball share is in line with LaLiga's published 53-56% of ball-in-play time.

---

## 6. Reproduction

- **Platform**: macOS 26 on Apple silicon (arm64), Python 3.14.4.
- **Dependencies**: `requirements-team.txt`, every package pinned, transitive ones included (pip
  freeze of the environment that produced the file): numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1,
  scipy 1.18.1, pillow 13.0.0 and the rest. A new virtual environment built from it alone gives the
  same output file, byte for byte, from the trained model.
- **External tools**: `ffmpeg` to read the tactical video's audio (whistle).
- **Data**: unpack the competition bundle into `data/` (or point `MTAH_DATA_DIR` at it). The
  external datasets are downloaded by the readers into `data/external/`:

```bash
python solution/external/idsse.py --download        # DFL / IDSSE, figshare
python solution/external/metrica.py --download      # Metrica sample data, GitHub
python solution/external/skillcorner.py --download  # SkillCorner open data, GitHub
python solution/external/pff.py --download          # PFF World Cup 2022, Google Drive
                                                    # (resumable: Drive throttles after
                                                    # about 40 files in a row)
```

- **Generation**:

```bash
python solution/team_reassign.py train                       # team model, external matches only
python solution/team_reassign.py apply                       # BAR-ATH teams re-decided
python solution/possession_ml.py train3                      # final three-state model
python solution/possession_ml.py predict3 --tracking tactical_clean
python evaluation/validate_submission.py evaluation/results/v2cml3_possession.csv
```

- **Determinism**: all models use `random_state=0`, and the degradations use fixed seeds per
  match and condition.
- **Output**: `evaluation/results/v2cml3_possession.csv`, 199,319 rows, SHA-256
  `eeb2d401cd6b2106d89e8e1896c949572009b55595694b91ec35e732fa64169d`. The training was run once
  for this file; a second training from a clean clone was not run (it takes about 1.5 h).
- **Time**: about 1 h for the external downloads and the first parse; `train3` about 1.5 h on
  5 processes (Apple M5; 4.8 million rows, doubled by the team swap); prediction about a minute.

---

## 7. Verification and limitations

**Checks:**
- `validate_submission.py` on the exact file submitted.
- The public window scored only after each decision.
- Frame counts, halves and the share of `X` compared with `games.py`.
- Review videos of the whole match showing the tactical video, the 2D pitch, the model's
  probabilities and the labels (`viz/v2_side_by_side.py`), used to find defects. The defects were
  then fixed in general and measured on the external matches.

**Known failure modes:**
- **Provider shift is the main risk.** Trained on three datasets and tested on the fourth, every
  model loses a lot (PFF 0.85 → 0.74). Nothing guarantees that BAR-ATH's tracking behaves like any
  of the four.
- **Short possessions.** Rebounds and quick exchanges under 2 s are mostly merged. This is where
  the oracle says most of the remaining points are (perfect team within 3 s of true changes:
  0.703 → 0.848).
- **Keeper holds are reduced, not solved.** On Metrica, the closest dataset to BAR-ATH, 32% of
  hold frames are still dead. On BAR-ATH 44% of the frames of the 8 probable holds are labelled
  dead, and 2 holds are wholly dead. The camera-framed training trades some of this for the set
  pieces (section 4, stage 7).
- **Penalties are rare.** 3 true penalties in the DFL matches, none labelled in the PFF events we
  have. Out of fold the final penalty rule is even with the previous one, so its value on BAR-ATH
  rests on the two cases it reads there, which we could only check from the players' layout.
- **The late additions are even on visible-ball tracking.** Keeper and set-piece work move the
  realistic score from 0.703 to 0.704. Their gains are in the situations they target (holds called
  dead, camera-framed tracking), which are frequent on BAR-ATH but rare in the external data.
- **Short stoppages with the ball untracked** (for example a 5.5 s one in the public window) can
  be missed. Long ball gaps are read as stoppages more readily than on the training data, where
  gaps are almost always dead.
- **Restart timing** in gaps depends on when the tracker finds the ball again.

**With more time:**
- Signals relative to the match itself (ratios, ranks) that survive the provider test.
- Re-detecting the ball, validated on external video with known truth.
- Using the broadcast where the tactical camera loses the ball.
- An estimate of the ball's height from the two camera projections.
- A decoder that allows short possessions after a fast ball deflection.
