# VisionGuard RUNBOOK (step by step)
Replace `<user>` with your DagsHub username and `<token>` with your DagsHub access token.
Take every screenshot with your terminal prompt/username, date/time and project name visible.
Windows PowerShell = training and data work. Ubuntu VM = Docker work (T6).

## 0. One-time accounts
1. GitHub: create empty repo `23L-0808_A2_Ops` (public).
2. DagsHub: New Repository -> Connect a repo -> pick the GitHub repo. Name it `23L-0808_A2_Ops`.
3. DagsHub: Remote -> Experiments shows the MLflow URI `https://dagshub.com/<user>/23L-0808_A2_Ops.mlflow`. User Settings -> Tokens -> create a token.
4. Kaggle: Account -> Create New API Token (kaggle.json) -> place in `C:\Users\<you>\.kaggle\kaggle.json`.

## T1.1 Environment (needs Python 3.10-3.12; 3.11 is safest)
```powershell
cd <unzipped folder>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt kaggle
python --version
pip list | findstr /i "tensorflow keras mlflow numpy pandas scikit-learn dvc"
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```
Screenshot: the versions and the GPU line (an empty list `[]` is expected on Windows CPU; say so).
```powershell
copy .env.example .env      # then edit .env: set the 3 MLFLOW_* values
git init; git branch -M main
git remote add origin https://github.com/<github-user>/23L-0808_A2_Ops.git
git check-ignore -v .env    # screenshot: proves .env is git-ignored (T3.2)
```

## T1.2 / 1.3 Data audit and EDA
```powershell
kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images -p data/raw --unzip
python -m src.audit data/raw        # screenshot the printed tables; saved to reports/
python -m src.eda                   # figures in reports/ -> write 2-3 sentences each
python -m src.splits                # creates splits/train|val|test.csv
```
If class folders are nested twice (`AiArtData/AiArtData`) it still works, since the class is the parent folder name.

## T1.4 DVC
**Toy part** (separate empty folder):
```powershell
mkdir ..\dvc-toy; cd ..\dvc-toy
git init; dvc init
"a,b`n1,2`n3,4" | Out-File -Encoding ascii data.csv
dvc add data.csv; git add data.csv.dvc .gitignore; git commit -m "v1 data"
"5,6" | Add-Content data.csv
dvc add data.csv; git add data.csv.dvc; git commit -m "v2 data"
git log --oneline
git checkout HEAD~1 -- data.csv.dvc; dvc checkout      # back to v1 (screenshot file contents)
mkdir ..\dvc-remote; dvc remote add -d local ..\dvc-remote
dvc push; del data.csv; dvc pull                        # data comes back
```
**Apply** (in the project folder):
```powershell
cd ..\23L-0808_A2_Ops
dvc init
dvc add data/raw
git add data/raw.dvc data/.gitignore .dvc .gitignore
git commit -m "Track dataset with DVC"
dvc remote add -d origin https://dagshub.com/<user>/23L-0808_A2_Ops.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user>
dvc remote modify origin --local password <token>
dvc push
git add .; git commit -m "VisionGuard scaffold + splits"; git push -u origin main
```
Write-up: the `.dvc` file holds the MD5 hash, size and path of the data. Git stores that tiny pointer and DVC stores the data in remote storage, because Git is slow and bloats with large binary files and keeps every version forever.

## T2 Models (run on a small trial first: `--epochs 2 --ft-epochs 1`)
```powershell
python -m src.train --model cnn --run-name smoke --epochs 2     # quick check everything works
```
Evidence: `model.summary()` output (screenshot), `out/<run>/trainable_stage1.txt` and `trainable_stage2.txt`, `confusion_matrix.png`, `roc_curve.png`, `misclassified.png`. Best epoch is logged as `best_epoch_stage1/2`. Take the augmented-batch figure yourself:
```powershell
python -c "from src.data import make_ds; import matplotlib.pyplot as plt; ds=make_ds('splits/train.csv',224,16,True); x,y=next(iter(ds)); [ (plt.subplot(4,4,i+1), plt.imshow(x[i].numpy()/255), plt.axis('off')) for i in range(16)]; plt.savefig('reports/aug_batch.png')"
```
Keras Tuner:
```powershell
python scripts/toy_tuner.py              # toy table (T2.7)
python scripts/tune_visionguard.py       # 8 trials; slow on CPU, lower epochs/sample if needed
```
Discussion points: learning rate usually matters most; picking the best validation score over many trials overfits the validation set, so the test score is what you trust.

## T3 MLflow
**3.1 local toy:**
```powershell
Remove-Item Env:MLFLOW_TRACKING_URI -ErrorAction SilentlyContinue   # make sure it's local
# temporarily use a .env with the MLFLOW_* lines commented out
python scripts/toy_mlflow.py --lr 0.001
python scripts/toy_mlflow.py --lr 0.01
mlflow ui          # open http://127.0.0.1:5000 (stop Flask first), tick both runs, click Compare
```
**3.2 DagsHub:** restore the `.env` values and repeat `python scripts/toy_mlflow.py --lr 0.001`, then screenshot the DagsHub Experiments tab.

**3.3-3.5 VisionGuard runs (8 runs = 6+ required):**
```powershell
.\scripts\run_experiments.ps1
```
r8 uses autolog and the others use the manual callback (3.3). In DagsHub -> Experiments select runs -> Compare for the table (3.5).
Explain: manual logging captures exactly what you choose (custom tags, test metrics, threshold), while autolog captures params, per-epoch metrics and the model summary automatically but misses your custom tags and test-set metrics. Fill any gaps by checking the run page.

**3.6 Registry:** register the best run twice, so two versions exist (e.g. best run, then the second-best run):
```powershell
python -m src.register <run_id_best>                       # v1
python -m src.register <run_id_best> --alias champion      # v2, alias champion
python -m src.load_champion path\to\some_image.jpg         # screenshot terminal output
python -m src.gradcam out/<best_run_name>/keras_model/best.keras --mlflow   # bonus
```
(Pick a different run for v1 if you want the versions to differ.) Screenshot the DagsHub model registry showing the `champion` alias.

## T4/T5 Run the Flask app locally (needs models/best.keras from `fetch_model`)
```powershell
python -m src.fetch_model
python -m app.app            # http://127.0.0.1:5000 ; SQLite file visionguard.db is the DB
```
Screenshots: home page, result page, history page, `/health` JSON, and the three failures: upload `notes.txt` (wrong type), a file > 5 MB (too large), and a text file renamed to `fake.png` (cannot decode).

## T6 Docker (Ubuntu VM)
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && newgrp docker
git clone https://github.com/<github-user>/23L-0808_A2_Ops.git && cd 23L-0808_A2_Ops
```
**6.1 warm-up** (screenshot each):
```bash
docker run -d --name web -p 8080:80 -v $PWD:/usr/share/nginx/html:ro nginx   # run, map port, mount volume
curl localhost:8080; docker logs web; docker exec -it web sh                  # logs, shell
docker ps -a; docker stop web; docker rm web; docker images; docker rmi nginx # list and remove
```
**Model into the VM:** copy `models/best.keras` and `models/meta.json` from Windows (WinSCP, `scp`, or a VMware shared folder) into `./models/` (it's git-ignored on purpose).
```bash
cp .env.example .env && nano .env      # set POSTGRES_PASSWORD, MODEL_PATH=/models/best.keras
docker compose up -d --build
docker compose ps                       # screenshot: db healthy, web healthy
docker compose exec web whoami          # appuser (non-root proof)
nc -zv localhost 5432                   # should FAIL: DB port is closed to the outside
docker image ls; docker history visionguard-a2_ops-web   # sizes and largest layers (6.4)
curl localhost:5000/health
```
**6.4 optimization:** record the size, then change something (the Dockerfile already uses `--no-cache-dir` and slim, so for a clear before/after temporarily remove `PIP_NO_CACHE_DIR`, `.dockerignore`, or switch from `tensorflow-cpu` to `tensorflow`), rebuild and compare the two sizes.
**6.5 operations and break-it:**
```bash
docker compose stop; docker compose start; docker compose restart web
docker compose logs -f web; docker compose exec db sh
# break: set a wrong POSTGRES_PASSWORD in .env only for web, or rename models/best.keras
docker compose up -d --force-recreate; docker compose logs web --tail 30   # diagnose from logs, then fix
```
**6.6 GitHub Actions:** toy repo with `.github/workflows/t.yml` (`on: push`, matrix `python-version: ["3.10","3.11"]`, step `pytest`) and a `test_x.py` containing `assert 1 == 2` -> red run -> fix -> green. For the project, `ci.yml` is already included. Push and add the badge (README has it). Secrets live in repo Settings -> Secrets and variables -> Actions (encrypted, masked in logs); a workflow file is committed to git and visible to everyone.

## T7 Deploy the model API (Pathway C)
The API image is the same Dockerfile; it downloads the champion from DagsHub once at start-up (never per request).
**Option 1, Hugging Face Spaces (free CPU, enough RAM for TensorFlow):**
1. New Space -> SDK: Docker -> blank. Clone it, copy `Dockerfile`, `requirements-app.txt`, `src/`, `api/` into it.
2. Edit the Space `Dockerfile` last lines: `EXPOSE 7860` and `CMD sh -c "python -m src.fetch_model && python -m api.server"`; remove the HEALTHCHECK line or point it at port 7860.
3. In the Space `README.md` front matter add `app_port: 7860`.
4. Space Settings -> Variables and secrets (secrets): `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MODEL_API_KEY`, and variable `PORT=7860`, `MODEL_PATH=models/best.keras`.
5. `git push`; wait for "Running". API URL: `https://<user>-<space>.hf.space`. Verify the free-tier terms and port in the current docs before relying on them.

**Option 2, Render/Railway:** New Web Service from the GitHub repo (Docker), start command `sh -c "python -m src.fetch_model && python -m api.server"`, same env vars. Free 512 MB RAM may be too small for TensorFlow, so check.

**Postman (from a different network, e.g. mobile hotspot):**
- `GET <API_URL>/health` -> 200.
- `POST <API_URL>/predict`, Headers `X-API-Key: <key>`, Body -> form-data -> key `file` (type File). Send 2 images; also send one without the key to show `401`. Screenshot each plus the JSON.

## T8 Integrate Flask with the cloud API
```powershell
# .env
MODEL_API_URL=https://<your-api-url>
MODEL_API_KEY=<same key>
python -m app.app        # no local model needed now
```
Upload 5+ different images, screenshot each result, then screenshot `/history` showing them (db rows come from the local DB, via the remote prediction). `/health` should show `"mode": "remote"`.

## Submission
- Report `23L-0808_VisionGuard.pdf` with front matter (name, roll no, GitHub/DagsHub/live links, commit hash `git rev-parse HEAD`, AI-assistant disclosure).
- DagsHub link, API URL, tested from another network.
