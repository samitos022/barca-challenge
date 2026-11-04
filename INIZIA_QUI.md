# Guida del team — More than a Hack 2027

Verificata il 4 novembre 2026. Le istruzioni ufficiali nei file originali prevalgono su questa guida.

## Cosa dobbiamo costruire

Un algoritmo che, per ciascun fotogramma del video tattico Barcelona–Athletic Club, produca H (Barça in possesso), A (Athletic in possesso), D (palla non in gioco) oppure X (astensione). Output: CSV `frame,label`, 199.319 righe di dati, frame da 0 a 199318 inclusi.

Il possesso passa all'avversario quando acquisisce controllo; durante un pallone conteso resta alla squadra che lo controllava prima. Una palla non rilevata dal tracking non implica una palla fuori gioco.

La graduatoria usa spellF1: premia possessi ricostruiti come intervalli temporali. macroF1 valuta le singole etichette ed è il primo spareggio. Spezzare un possesso in tanti cambi artificiali penalizza il risultato. I punteggi locali sono sui soli cinque minuti pubblici, non sull'intera partita.

## Iscrizione e consegna

- Ogni partecipante si iscrive e accetta i termini: https://landing.mobileworldcapital.com/more-than-a-hack-2027
- Usare lo stesso nome del team; il regolamento richiede 3–5 membri e specifica che sotto tre non si è idonei a Innovation Challenge e Final Pitch.
- Scaricare il pacchetto tramite il link ricevuto dagli organizzatori; mantenerne le cartelle `tactical/` e `broadcast/` dentro `data/`, oppure impostare `MTAH_DATA_DIR`.
- Entro il **17 novembre 2026** consegnare CSV, URL della repository pubblica e relazione tecnica breve (Markdown o PDF). Il modulo arriverà via email. Verificare nell'email l'eventuale orario/fuso della scadenza.
- Un invio per team, modificabile; serve un account Google. Nominare CSV e relazione con il nome del team. Non c'è una classifica live.

## Avvio locale

Da questa directory, su Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-team.txt
python evaluation/run_evaluation.py
python evaluation/validate_submission.py evaluation/results/baseline_smooth50.csv
```

Windows PowerShell: creare l'ambiente con `py -m venv .venv` e attivarlo con `.venv\Scripts\Activate.ps1`.

I quattro CSV baseline sono già inclusi: questi controlli non richiedono i dati riservati. Solo dopo aver scaricato i dati, `python starter/baseline.py` rigenera le baseline dal tracking. Per i notebook, installare anche `requirements-starter.txt` e Jupyter; partire da `starter/03_first_submission.ipynb`.

Su questa macchina la rete di pip non funzionava: `.venv` è stato configurato con `--system-site-packages` per usare numpy 1.26.4 e pandas 2.1.4 già presenti. Per gli altri computer usare l'installazione normale sopra. `requirements-team.txt` fissa le due dipendenze dirette di partenza; prima della consegna fissare anche tutte le dipendenze transitive e gli strumenti effettivamente usati, verificando una nuova installazione pulita.

## Una sola repository GitHub per tutti

Consiglio una repository autonoma del team, inizialmente privata se preferite, pubblica alla consegna. Basta l'account personale di un membro; un'organizzazione non è necessaria. Non serve una repository per ciascun amico.

Questa cartella è una nuova repository locale: importata dall'archivio ufficiale, senza la cronologia originale. Il remote `upstream` identifica la fonte; `origin` punta a `https://github.com/samitos022/barca-challenge.git`. Non fare un merge automatico con upstream: le cronologie sono separate; confrontare e riportare eventuali aggiornamenti ufficiali.

1. Repository indicata dal team: https://github.com/samitos022/barca-challenge. Il remote locale è già configurato. Prima del primo push verificare che la repository remota sia vuota; se contiene commit, integrarli senza usare force push.
2. Dopo aver controllato `git status` e i file da pubblicare:

```bash
git add .
git commit -m "Initialize team workspace from official starter"
git push -u origin main
```

Se si usa GitHub CLI, prima ripristinare l'accesso con `gh auth login -h github.com`; non incollare token in documenti o chat.

3. In GitHub: Settings → Collaborators → Add people. Il proprietario invita gli amici tramite username; ciascuno accetta l'invito e clona **la repository del team**.
4. Per ogni attività creare un branch, ad esempio `git switch -c feature/temporal-possession`; push e Pull Request verso `main`, controllata da un compagno.
5. Tenere `main` eseguibile. Usare Issues per dividere le attività. Prima dell'invio rendere la repository pubblica e verificarne l'accesso senza login.

Repository indicata: `samitos022/barca-challenge`. Pubblicazione non ancora verificata: Git non riesce a raggiungere github.com da questo ambiente e la richiesta API pubblica restituisce 404 (può trattarsi di una repository privata). Gli inviti ai collaboratori non sono stati inviati; il nome del team e i collaboratori restano da definire.

## Piano di lavoro proposto

1. Tutti: iscrizione, lettura delle regole e avvio dello scorer.
2. Una persona: caricamento tracking, coordinate e allineamento dei frame tattici. Il broadcast ha una timeline diversa.
3. Una persona: stimare possesso con continuità temporale, gestione delle occlusioni e controllo effettivo della palla.
4. Una persona: distinguere gioco attivo/interruzioni usando segnali automatici, eventualmente Smart Tagging.
5. Integrare in un comando riproducibile; registrare modifiche, motivazioni e dipendenze nella relazione.
6. Riservare gli ultimi giorni a riproduzione da ambiente pulito, validazione CSV, pubblicazione e consegna.

Questa è una proposta tecnica, non un requisito degli organizzatori. Non è ancora implementata una soluzione del team.

## Regole che influenzano il metodo

- Vietato annotare o correggere manualmente questa partita, anche solo per addestramento o validazione; niente regole ad hoc legate a frame osservati a mano.
- I cinque minuti pubblici servono a verificare formato, etichette e allineamento: vietato usarli per training o tuning.
- Niente dati umani esterni su questa stessa partita. Dati di altre partite sono ammessi se accessibili agli organizzatori e documentati.
- Non redistribuire video, tracking o derivati che permettano di ricostruirli. Tenerli in `data/` (ignorata da Git) o fuori dalla repo. Anche i notebook possono incorporare dati nei loro output: ripulirli prima del commit.
- OVERVIEW.md vieta l'invio di materiali riservati, anche estratti, a LLM ospitati da terzi/cloud. L'uso di LLM per il codice è ammesso; non caricare qui i dati della gara.
- Il codice deve rigenerare esattamente il CSV consegnato.

## Fonti

Repository: https://github.com/metrica-sports/more-than-a-hack-2027

Leggere nell'ordine: [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md), [RULES.md](RULES.md), [FAQ.md](FAQ.md), [OVERVIEW.md](OVERVIEW.md).

Origine dell'importazione: archivio GitHub di `main`, prefisso commit `f8f2629`, scaricato il 4 novembre 2026. Licenza originale MIT conservata in LICENSE; non comprende i dati della partita.
