#!/bin/sh

# embedding compute -i 1000 --checkpoint 1 --scale 0 -o output/pi
# embedding compute -i 100000 --checkpoint 100 --scale 0 -o output/pi
# 
# # 50th eigenvalue: 393.15
# # Using 390 ^ 2 / 4
# embedding compute -i 1000 --checkpoint 1 --scale 0 --beta 38025 -o output/pim
# 
# Distinct Words: 71290
# Number of non-zeros: 60666466

embedding compute --solver alecton --scheme element --batch 606665 --eta 0.0001 -i 25000 --checkpoint 100 --normfreq 100 --scale 0 -o output/alecton.ele.606665.0001
embedding compute --solver alecton --scheme row     --batch 713    --eta 0.0001 -i 25000 --checkpoint 100 --normfreq 100 --scale 0 -o output/alecton.row.713.0001
embedding compute --solver alecton --scheme column  --batch 713    --eta 0.0001 -i 25000 --checkpoint 100 --normfreq 100 --scale 0 -o output/alecton.col.713.0001

embedding compute --solver alecton --scheme element --batch 606665 --eta 0.00001 -i 25000 --checkpoint 100 --normfreq 100 --scale 0 -o output/alecton.ele.606665.00001
embedding compute --solver alecton --scheme row     --batch 713    --eta 0.00001 -i 25000 --checkpoint 100 --normfreq 100 --scale 0 -o output/alecton.row.713.00001
embedding compute --solver alecton --scheme column  --batch 713    --eta 0.00001 -i 25000 --checkpoint 100 --normfreq 100 --scale 0 -o output/alecton.col.713.00001

embedding compute --solver alecton --scheme element --batch 606665 --eta 0.000001 -i 25000 --checkpoint 100 --normfreq 100 --scale 0 -o output/alecton.ele.606665.000001
embedding compute --solver alecton --scheme row     --batch 713    --eta 0.000001 -i 25000 --checkpoint 100 --normfreq 100 --scale 0 -o output/alecton.row.713.000001
embedding compute --solver alecton --scheme column  --batch 713    --eta 0.000001 -i 25000 --checkpoint 100 --normfreq 100 --scale 0 -o output/alecton.col.713.000001


embedding compute --solver alecton --scheme element --batch 6066647 --eta 0.001 -i 2500 --checkpoint 10 --normfreq 10 --scale 0 -o output/alecton.ele.6066647.001
embedding compute --solver alecton --scheme row     --batch 7129    --eta 0.001 -i 2500 --checkpoint 10 --normfreq 10 --scale 0 -o output/alecton.row.7129.001
embedding compute --solver alecton --scheme column  --batch 7129    --eta 0.001 -i 2500 --checkpoint 10 --normfreq 10 --scale 0 -o output/alecton.col.7129.001

embedding compute --solver alecton --scheme element --batch 6066647 --eta 0.0001 -i 2500 --checkpoint 10 --normfreq 10 --scale 0 -o output/alecton.ele.6066647.0001
embedding compute --solver alecton --scheme row     --batch 7129    --eta 0.0001 -i 2500 --checkpoint 10 --normfreq 10 --scale 0 -o output/alecton.row.7129.0001
embedding compute --solver alecton --scheme column  --batch 7129    --eta 0.0001 -i 2500 --checkpoint 10 --normfreq 10 --scale 0 -o output/alecton.col.7129.0001

embedding compute --solver alecton --scheme element --batch 6066647 --eta 0.00001 -i 2500 --checkpoint 10 --normfreq 10 --scale 0 -o output/alecton.ele.6066647.00001
embedding compute --solver alecton --scheme row     --batch 7129    --eta 0.00001 -i 2500 --checkpoint 10 --normfreq 10 --scale 0 -o output/alecton.row.7129.00001
embedding compute --solver alecton --scheme column  --batch 7129    --eta 0.00001 -i 2500 --checkpoint 10 --normfreq 10 --scale 0 -o output/alecton.col.7129.00001


embedding compute --solver alecton --scheme element --batch 30333233 --eta 0.005 -i 500 --checkpoint 2 --normfreq 2 --scale 0 -o output/alecton.ele.30333233.005
embedding compute --solver alecton --scheme row     --batch 35645    --eta 0.005 -i 500 --checkpoint 2 --normfreq 2 --scale 0 -o output/alecton.row.35645.005
embedding compute --solver alecton --scheme column  --batch 35645    --eta 0.005 -i 500 --checkpoint 2 --normfreq 2 --scale 0 -o output/alecton.col.35645.005

embedding compute --solver alecton --scheme element --batch 30333233 --eta 0.0005 -i 500 --checkpoint 2 --normfreq 2 --scale 0 -o output/alecton.ele.30333233.0005
embedding compute --solver alecton --scheme row     --batch 35645    --eta 0.0005 -i 500 --checkpoint 2 --normfreq 2 --scale 0 -o output/alecton.row.35645.0005
embedding compute --solver alecton --scheme column  --batch 35645    --eta 0.0005 -i 500 --checkpoint 2 --normfreq 2 --scale 0 -o output/alecton.col.35645.0005

embedding compute --solver alecton --scheme element --batch 30333233 --eta 0.00005 -i 500 --checkpoint 2 --normfreq 2 --scale 0 -o output/alecton.ele.30333233.00005
embedding compute --solver alecton --scheme row     --batch 35645    --eta 0.00005 -i 500 --checkpoint 2 --normfreq 2 --scale 0 -o output/alecton.row.35645.00005
embedding compute --solver alecton --scheme column  --batch 35645    --eta 0.00005 -i 500 --checkpoint 2 --normfreq 2 --scale 0 -o output/alecton.col.35645.00005


for eta in 001 0001 00001 000001
do
    for beta in 0 38025
    do
        embedding compute --solver vr --scheme element --batch 606665 --eta 0.${eta} --beta ${beta} --innerloop 100 -i 250 --checkpoint 1 --normfreq 1 --scale 0 -o output/vr.ele.606665.${eta}.${beta}
        embedding compute --solver vr --scheme row     --batch 713    --eta 0.${eta} --beta ${beta} --innerloop 100 -i 250 --checkpoint 1 --normfreq 1 --scale 0 -o output/vr.row.713.${eta}.${beta}
        embedding compute --solver vr --scheme column  --batch 713    --eta 0.${eta} --beta ${beta} --innerloop 100 -i 250 --checkpoint 1 --normfreq 1 --scale 0 -o output/vr.col.713.${eta}.${beta}
    done
done

for eta in 01 001 0001 00001
do
    for beta in 0 38025
    do
        embedding compute --solver vr --scheme element --batch 6066647 --eta 0.${eta} --beta ${beta} --innerloop 10 -i 250 --checkpoint 1 --normfreq 1 --scale 0 -o output/vr.ele.6066647.${eta}.${beta}
        embedding compute --solver vr --scheme row     --batch 7129    --eta 0.${eta} --beta ${beta} --innerloop 10 -i 250 --checkpoint 1 --normfreq 1 --scale 0 -o output/vr.row.7129.${eta}.${beta}
        embedding compute --solver vr --scheme column  --batch 7129    --eta 0.${eta} --beta ${beta} --innerloop 10 -i 250 --checkpoint 1 --normfreq 1 --scale 0 -o output/vr.col.7129.${eta}.${beta}
    done
done

for eta in 05 005 0005 00005
do
    for beta in 0 38025
    do
        embedding compute --solver vr --scheme element --batch 30333233 --eta 0.${eta} --beta ${beta} --innerloop 2 -i 250 --checkpoint 1 --normfreq 1 --scale 0 -o output/vr.ele.30333233.${eta}.${beta}
        embedding compute --solver vr --scheme row     --batch 35645    --eta 0.${eta} --beta ${beta} --innerloop 2 -i 250 --checkpoint 1 --normfreq 1 --scale 0 -o output/vr.row.35645.${eta}.${beta}
        embedding compute --solver vr --scheme column  --batch 35645    --eta 0.${eta} --beta ${beta} --innerloop 2 -i 250 --checkpoint 1 --normfreq 1 --scale 0 -o output/vr.col.35645.${eta}.${beta}
    done
done
