python train_search.py --save AID_20 --data ../data/AID_20 --num_class 30  --epochs 1
python train_search.py --save AID_50 --data ../data/AID_50 --num_class 30  --epochs 1
python train_search.py --save NWPU10 --data ../data/NWPU_10 --num_class 45  --epochs 1
python train_search.py --save NWPU20 --data ../data/NWPU_20 --num_class 45  --epochs 1
python train_search.py --save leaf   --data ../data/leaf --num_class 38  --epochs 1

python train_search.py --save nas_eval_hard_flower   --data ../data/nas_eval_hard_flower --num_class 102  --epochs 1
python train_search.py --save nas_eval_hard_indoor   --data ../data/nas_eval_hard_indoor --num_class 45  --epochs 1
python train_search.py --save nas_eval_hard_sport    --data ../data/nas_eval_hard_sport --num_class 8  --epochs 1

pause