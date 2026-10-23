\# WEX 428 — Handwritten Math Symbol AI



Full-stack ML platform for handwritten mathematical symbol recognition,

comparing a baseline CNN, an augmented CNN, and a self-supervised +

domain-adversarial model. Includes handwritten formula recognition

(multi-symbol segmentation + reconstruction).



**Live app:** https://handwritten-math-symbol-ai.onrender.com



\## Repo structure





\## Local development



Backend:

```bash

cd backend

python -m venv venv \&\& source venv/bin/activate   # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

uvicorn app.main:app --reload

```

Visit `http://127.0.0.1:8000/docs` to confirm it's running.



Frontend:

```bash

cd frontend

python -m http.server 5500

```

Visit `http://localhost:5500` — `app.js` automatically points at `127.0.0.1:8000`

when running locally.



Or run both together: `docker compose up --build`.



\## Deployment



\*\*Backend (Render, Docker environment):\*\*

1\. New Web Service → connect this repo.

2\. Environment: Docker (Render auto-detects `backend/Dockerfile`).

3\. \*\*Service name must be `workplace428`\*\* so its URL (`workplace428.onrender.com`)

&#x20;  matches what `frontend/app.js` already expects. If you use a different name,

&#x20;  update `API\_BASE` in `app.js` to match instead.

4\. Environment variables:

&#x20;  - `SECRET\_KEY` — any long random string (used for JWT signing)

&#x20;  - `ALLOWED\_ORIGINS` — `https://sadik-2025.github.io` (tightens CORS)

5\. Deploy. Confirm `/health` returns `{"status": "ok"}`.



\*\*Frontend (GitHub Pages):\*\* already live — deploys automatically on push to `main`.



\*\*Note:\*\* Render's free tier uses ephemeral disk, so the SQLite database

(`backend/app.db`) resets on redeploy/restart. Fine for demos; for persistence,

switch `DATABASE\_URL` to a managed Postgres instance.



\## Model comparison



Three approaches, trained and evaluated in `ml/scripts/`, metrics saved to

`ml/models/<name>/metrics.json` and served via `/models/compare`:



\- \*\*baseline\*\* — plain CNN

\- \*\*augmented\*\* — same CNN, trained with data augmentation

\- \*\*ssl\_domain\_gen\*\* — self-supervised pretraining + domain-adversarial fine-tuning (default/recommended model in the UI)



\## Known limitations



\- SQLite resets on Render free-tier redeploys (see above).

\- Formula recognition segments symbols via classical CV contour detection —

&#x20; works best with clear spacing between symbols; touching/overlapping

&#x20; symbols can merge into one segment.

