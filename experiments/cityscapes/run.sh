mkdir -p ./save
mkdir -p ./trainlogs

method=NURBS
num_bends=5
seed=0
lr=1.e-4


python3 trainer.py  --seed=$seed --num_bends=$num_bends --lr=$lr > trainlogs/$method-num_bends$num_bends-sd$seed.log 2>&1 &
