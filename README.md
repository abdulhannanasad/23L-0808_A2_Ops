# VisionGuard (23L-0808_A2_Ops)
![CI](https://github.com/abdulhannanasad/23L-0808_A2_Ops/actions/workflows/ci.yml/badge.svg)

## Run order (PowerShell)
```
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env            # fill in DagsHub values, never commit
kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images -p data/raw --unzip
python -m src.audit data/raw; python -m src.eda; python -m src.splits
git init; dvc init; dvc add data/raw; git add data/raw.dvc .gitignore; git commit -m "track data"
dvc remote add -d origin https://dagshub.com/<user>/23L-0808_A2_Ops.dvc
dvc remote modify origin --local auth basic; dvc remote modify origin --local user <user>; dvc remote modify origin --local password <token>
dvc push
.\scripts\run_experiments.ps1
python -m src.register <best_run_id> --alias champion   # run twice (two versions) before aliasing
python -m src.fetch_model
```
## Serve
```
docker compose up --build     # http://localhost:5000
python -m api.server          # model API (deploy this image/command to the cloud)
```
