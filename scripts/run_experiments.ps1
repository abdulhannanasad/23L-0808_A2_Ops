# Task 3.5: six+ runs, each differing by ONE change from a reference (PowerShell)
python -m src.train --model cnn       --run-name r1_cnn_baseline
python -m src.train --model mobilenet --run-name r2_mnv2_lr1e-3 --lr 1e-3
python -m src.train --model mobilenet --run-name r3_mnv2_lr1e-4 --lr 1e-4
python -m src.train --model mobilenet --run-name r4_mnv2_noaug   --aug off
python -m src.train --model mobilenet --run-name r5_mnv2_frozen  --no-finetune
python -m src.train --model mobilenet --run-name r6_mnv2_img160  --img-size 160
python -m src.train --model efficientnet --run-name r7_effb0
python -m src.train --model mobilenet --run-name r8_mnv2_autolog --logging autolog
