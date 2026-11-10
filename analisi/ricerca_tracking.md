# Rifare il tracking: opzioni pubbliche (ricerca del 6/11/2026)

Sintesi in LAVORO_SVOLTO.md, sezione 6.9. Fonti principali:
- Roboflow sports (YOLOv8 4 classi ball/goalkeeper/player/referee, palla a tasselli, ByteTrack, squadre SigLIP+UMAP+KMeans, MPS supportato; codice MIT, pesi AGPL): https://github.com/roboflow/sports/tree/main/examples/soccer
- WASB-SBDT, palla (MIT; pesi "Soccer" su ISSIA-CNR, vista larga fissa 1080p: F1 88): https://github.com/nttcom/WASB-SBDT , https://arxiv.org/html/2311.05237v2
- SoccerNet-v3D, YOLOv11 palla (yolo-issia-ball-opt per vista larga; GPL-2.0): https://github.com/mguti97/SoccerNet-v3D
- Roboflow trackers (ByteTrack, OC-SORT, BoT-SORT, McByte; Apache 2.0): https://github.com/roboflow/trackers
- sn-gamestate / TrackLab (baseline SoccerNet GSR; GPL-3.0, pensato per CUDA): https://github.com/SoccerNet/sn-gamestate
- SoccerNet GSR 2025 (vincitori: YOLOX/RF-DETR + Deep-EIoU/BoT-SORT, squadre per colore, portieri per posizione): https://arxiv.org/html/2508.19182v1 ; Constructor Tech (5 cluster, portieri nelle aree): https://arxiv.org/html/2504.06357v1
- Banchi leciti con verità su altre partite: SoccerTrack v2 (10 partite intere panoramiche 4K, CC BY 4.0): https://atomscott.github.io/SoccerTrack-v2/ ; SoccerNet-Tracking; ISSIA-CNR (palla). IDSSE e Metrica sample: tracking + eventi ma senza video.
- Licenze: Ultralytics AGPL-3.0 (ok con repo pubblica, da dichiarare): https://ultralytics.com/license ; RF-DETR N/S/M Apache 2.0.
Da misurare: fps reali sull'M5 (nessun benchmark pubblico), supporto MPS di WASB.
