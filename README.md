# Possesso palla frame per frame — More than a Hack 2027

Per ogni frame della partita BAR-ATH (199.319 frame, 25 fps) il nostro codice dice **chi ha la
palla**:

| Etichetta | Significato |
|---|---|
| `H` | possesso della squadra di casa (Barcelona) |
| `A` | possesso della squadra ospite (Athletic Club) |
| `D` | palla morta (gioco fermo) |
| `X` | nessuna risposta (fuori dai tempi di gioco) |

La classifica usa lo **spellF1**: conta quanti possessi interi riconosciamo, non quanti frame
indoviniamo. Contano molto la tempistica dei cambi (1 s di ritardo pesa) e il numero di possessi.
Regole, dati e metrica ufficiali: [CHALLENGE.md](CHALLENGE.md), [RULES.md](RULES.md),
[DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md).

---

## Avvio rapido

```bash
# predizione (scrive evaluation/results/v2_possession*.csv)
.venv/bin/python solution/possession_v2.py --no-whistle   # basta il tracking (ATD)
.venv/bin/python solution/possession_v2.py                # + fischio: serve il video tattico
.venv/bin/python solution/possession_v2.py --tracking fused   # + broadcast (vedi sotto)
.venv/bin/python solution/possession_v2.py --fix-teams    # + correzione squadre (sperimentale)

# punteggio sui 5 minuti pubblici (solo per controllare che il file sia giusto)
.venv/bin/python evaluation/run_evaluation.py

# calibrazione su partite esterne (prima volta: scarica i dati)
.venv/bin/python solution/external/metrica.py --download
.venv/bin/python solution/calibrate.py baseline
```

I dati della partita vanno in `data/tactical/` (non sono in git). Le cache e i file derivati
finiscono in `data/derived/`.

---

## Cosa fa ogni file

```
solution/
  possession_v2.py      ← IL METODO: tracking → etichette H/A/D per ogni frame
  possession_v1.py      ← feature di base (palla, distanze, tocchi) usate da v2
  tracking.py           ← legge l'ATD e lo converte in metri (con cache)
  clean_tracking.py     ← sposta i tracklet finiti nella squadra sbagliata (--fix-teams)
  whistle.py            ← cerca toni da fischietto nell'audio del video tattico
  sync_broadcast.py     ← allinea il video broadcast a quello tattico (orologio, audio, tracking)
  ocr_vision.swift      ← OCR dell'orologio usato da sync_broadcast.py (macOS)
  fusion.py             ← unisce tracking tattico e broadcast (75% / 25%)
  calibrate.py          ← misura v2 sulle partite esterne: baseline, ablate, sweep, tracking
  restart_timing.py     ← di quanto sbagliamo le riprese, per tipo (rimessa, punizione, ...)
  external/
    metrica.py          ← partite di esempio Metrica: tracking + etichette dagli eventi
    degrade.py          ← sporca i giocatori delle partite esterne come quelli di BAR-ATH
viz/v2_review.py        ← video di revisione: video + campo 2D + verità/v2 + grafici dei punteggi
viz/                    ← altri video di revisione (solo uso locale)
starter/                ← codice fornito dagli organizzatori (+ temporal_possession.py, nostro
                          primo esperimento con Viterbi, superato da v2)
evaluation/             ← scorer ufficiale e 5 minuti di ground truth pubblica
```

Il cuore è `possession_v2.predict()`: riceve gli array del tracking e restituisce le etichette,
senza leggere file. Lo stesso codice gira su BAR-ATH e sulle partite esterne.

---

## Come funziona `possession_v2`

### L'idea

Una **macchina a tre stati** (H, A, D) che scorre i frame uno alla volta. Ogni possibile cambio
di stato ha un suo **punteggio**, che sale con le evidenze. Quando un punteggio supera la soglia
lo stato cambia, e il cambio viene **datato all'inizio dell'evidenza**, non al frame in cui
diventiamo sicuri. Tutte le soglie sono grandezze fisiche (metri, m/s, secondi) e si trovano in
`PARAMS`, in cima al file.

```
            tocco / controllo della squadra avversaria
     H  ◄──────────────────────────────────────────►  A
      \                                              /
       \  palla fuori · giocatori fermi · buco lungo /
        ▼                                          ▼
                             D
              ripresa: primo tocco, palla che ricompare
```

### 1. Cambio di possesso (H ↔ A)

La squadra **senza** palla accumula punti; quando arriva a 1 prende il possesso.

| Evidenza | Quando | Punti |
|---|---|---|
| Controllo | un suo giocatore è il più vicino alla palla (entro 1–2,5 m), almeno 1 m più vicino dell'avversario, e la palla è lenta (sotto 4 m/s) | 2 al secondo |
| Tocco | la velocità della palla cambia di colpo (oltre 3,5 m/s) vicino a un suo giocatore | 0,5 |
| Verifica a 2 s | dopo il tocco: la squadra tiene davvero la palla? | ±0,5 |

- Con palla veloce (oltre 8 m/s, passaggio o tiro) il controllo non conta: solo un tocco può rubarla.
- La squadra in possesso, con i propri controlli e tocchi, toglie punti all'altra.
- Senza nuove evidenze i punti si dimezzano ogni secondo.

### 2. Palla morta (H/A → D)

| Regola | Quando |
|---|---|
| **Palla fuori** | la palla supera una linea di oltre 1 m: palla morta subito, datata al passaggio della linea |
| **Buco lungo** (regola 8) | palla non tracciata per almeno 5 s **e** giocatori che camminano (velocità media sotto 1,55 m/s) **e** almeno un altro segnale: palla persa vicino a una linea o ricomparsa su una linea / in un punto da calcio piazzato, giocatori che frenano o lontani dalla palla, fischio. Allora palla morta dal primo frame mancante |
| **Giocatori fermi** | combinazione di due segnali: i 4 giocatori più vicini sono lontani dalla palla (oltre 7–12 m) e i giocatori frenano di colpo (velocità dimezzata rispetto ai 4 s prima) |
| Fischio | aggiunge punti solo se poi i giocatori frenano (il pubblico fischia spesso) |

La regola 8 è quella che conta di più. Nel tracking di Metrica la palla manca quasi sempre
proprio a gioco fermo. Su BAR-ATH lo si vede senza etichette: nei buchi lunghi (41% dei frame) i
giocatori vanno a 1,0 m/s, contro 1,9 m/s quando la palla è visibile. Il buco però resta un
segnale debole, perché la palla si perde anche nelle mischie: da solo non basta, serve un secondo
segnale. Su BAR-ATH lo hanno 89 buchi su 102.

### 3. Ripresa del gioco (D → H/A)

- **Palla che ricompare** dopo un buco lungo, in campo o appena fuori (rimesse): il gioco riparte
  in quel frame, e la palla va alla squadra del giocatore più vicino.
- **Primo tocco** in campo dopo almeno 1 s di palla ferma: la palla va alla squadra di chi tocca.
- **Gioco di nuovo vivo** (palla in campo, giocatori che corrono) se il tocco non si vede.
- Il primo avvio di ogni tempo vale solo dal centro del campo (calcio d'inizio).

**Quando riparte (regola 9).** In tutti i casi sopra la ripresa viene *confermata* dalla regola,
ma *datata* al momento in cui la palla, dopo essere stata lenta (sotto 3 m/s per almeno 0,5 s),
parte: il frame più lento dell'ultimo secondo prima che superi i 3 m/s. In questo tracking una
palla "ferma" non è mai davvero ferma: sistemata o tenuta in mano si legge a 1–2,5 m/s. Su BAR-ATH
anticipa 34 riprese su 110, di 1,5 s in mediana.

### 4. Tocco lampo (regola 10)

Un tocco della squadra senza palla che la manda a più di 6 m/s in una direzione diversa di almeno
60° è un intercetto. Se poi quella squadra prende il possesso (entro 4 s, senza che l'altra
squadra tocchi o controlli la palla nel frattempo), il possesso viene datato dal tocco lampo e non
dal primo controllo.

### Opzioni

Ogni regola nuova ha un interruttore in `PARAMS` (`GAP_SUPPORT`, `RESTART_MOVE`, `LIGHTNING`),
usato da `calibrate.py ablate`.

- `--fix-teams`: su BAR-ATH una squadra ha 12 giocatori nel 6% dei frame, quasi sempre perché un
  giocatore dell'altra squadra è finito nella sua. `clean_tracking.py` sposta il tracklet il cui
  cambio di squadra porta entrambe più vicino a 11. Dimezza le violazioni, ma il guadagno
  misurato (+0.007) è sotto il rumore, quindi è spento di default.
- `--tracking fused`: usa anche il broadcast. Prima vanno lanciati `sync_broadcast.py` e
  `fusion.py`; servono i video.
- Regola del calcio d'angolo: esiste ma è spenta (`CORNER_RULE = 0`), perché con la regola 8
  faceva ripartire il gioco prima che la palla ricomparisse.

---

## Come calibriamo (senza overfittare)

I 5 minuti pubblici **non** si usano per scegliere parametri ([RULES.md](RULES.md)): servono solo
a controllare che il file sia corretto. Calibriamo invece su **altre partite**, come le regole
permettono.

1. **Dati esterni.** Le 3 partite di esempio di Metrica Sports, lo stesso fornitore del nostro
   tracking (`external/metrica.py`). Le etichette H/A/D si ricavano dagli eventi: palla morta
   dall'interruzione (fuori, fallo, fuorigioco, gol) alla ripresa. Un recupero conta come cambio
   di possesso solo se la squadra poi gioca la palla, cioè un possesso *controllato*.
2. **Stesso livello di sporcizia.** La palla manca già come da noi (59–68% visibile contro 59%). I
   giocatori Metrica invece sono perfetti: `external/degrade.py` li sporca finché il numero di
   giocatori per squadra ha lo stesso istogramma di BAR-ATH (11 nel 58%, 10 nel 22%, 12+ nel 6%).
   Ogni partita viene sporcata 2 volte: 6 run in tutto.
3. **Regola di decisione.** Si tiene una modifica solo se migliora quasi tutte le 6 run. Sotto
   ~0.02 di spellF1 è rumore.

```bash
.venv/bin/python solution/calibrate.py baseline   # parametri attuali
.venv/bin/python solution/calibrate.py ablate     # ogni regola spenta, una alla volta
.venv/bin/python solution/calibrate.py sweep      # ogni parametro ×0.5 ×0.75 ×1.33 ×2
.venv/bin/python solution/calibrate.py tracking   # giocatori puliti / sporchi / corretti
.venv/bin/python solution/restart_timing.py       # ritardo delle riprese per tipo
```

---

## Risultati

**Partite esterne** (spellF1 medio su 6 run):

| Versione | spellF1 |
|---|---|
| v2 di partenza | 0.550 |
| + regola 8 (buco lungo) | 0.659 |
| + angoli spenti | 0.671 |
| + supporto obbligatorio per la regola 8, regole 9 e 10 (**attuale**) | **0.663** |

L'ultimo passo costa 0.008 su Metrica, e lo accettiamo di proposito. Su Metrica ogni buco della
palla è palla morta, quindi chiedere un secondo segnale fa perdere qualche caso vero. Su BAR-ATH il
buco da solo è meno affidabile, e il supporto lo rende più robusto. Le regole 9 e 10 su Metrica
sono neutre: lì la palla a gioco fermo non è mai visibile, quindi non si può misurarle.

Sweep dei parametri (ognuno ×0.5, ×0.75, ×1.33, ×2): i valori attuali sono su un plateau. L'unica
variazione che migliora (soglie di velocità più alte) rende la regola 8 più permissiva e sfrutta
l'anomalia dei buchi Metrica: non adottata.

Quanto vale ogni regola (se la tolgo): tocchi −0.041, verifica a 2 s −0.035, frenata −0.022.
Sono neutre (±0.005) la regola della causa, la distanza dei giocatori e la durata minima della
palla morta.

**Attenzione.** Su Metrica i buchi della palla coincidono al frame con la palla morta. Su BAR-ATH
il segnale c'è, ma è più debole: alla ricomparsa la palla è in posizione da calcio piazzato il 39%
delle volte, contro il 64–74% su Metrica. Il guadagno reale della regola 8 sarà quindi
probabilmente minore di +0.11.

**5 minuti pubblici** (solo controllo): v2 attuale senza fischio spellF1 0.545, macroF1 0.830.
Miglior baseline ufficiale 0.424 / 0.581.

**Video di revisione** (`viz/v2_review.py`): video tattico + campo 2D + strisce verità/v2 + i due
grafici che decidono i cambi. Scrive `data/clips/BAR-ATH_v2_review_public.mp4` e un riepilogo
`_overview.png`. Serve il video tattico in `data/tactical/BAR-ATH_tactical.mp4`.

---

## Cosa manca

- Valutare v2 con fischio e con tracking `fused` (servono i video).
- Usare il replay del broadcast come segnale di palla morta.
- Smart Tagging: le finestre dei tag (~35 s) sono troppo larghe per datare le riprese; va capito
  dove cade la ripresa dentro la finestra prima di usarli per scegliere la squadra.
- Report tecnico: [docs/TECHNICAL_REPORT_TEMPLATE.md](docs/TECHNICAL_REPORT_TEMPLATE.md).
